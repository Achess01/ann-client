import {useEffect, useMemo, useRef, useState} from 'react'
import {Link} from 'react-router-dom'
import ForceGraph3D from 'react-force-graph-3d'
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Tooltip,
    Legend,
} from 'chart.js'
import {Line} from 'react-chartjs-2'
import {WS_BASE_URL} from '../config'
import type {ClientMessage, ServerMessage} from '../types'

ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Tooltip,
    Legend,
)

type GraphNode = {
    id: string
    layer: 'input' | 'hidden' | 'output'
    fx: number
    fy: number
    fz: number
    activation: number
    error?: number
}

type GraphLink = {
    source: string
    target: string
    weight: number
}

const INPUT_COUNT = 784
const HIDDEN_COUNT = 64
const OUTPUT_COUNT = 10

const buildNodes = () => {
    const nodes: GraphNode[] = []

    const rowSize = 28;
    const spacing = 4; // Spacing between pixels
    for (let i = 0; i < INPUT_COUNT; i += 1) {
        const row = Math.floor(i / rowSize);
        const col = i % rowSize;
        nodes.push({
            id: `input-${i}`,
            layer: 'input',
            fx: -150,
            fy: (row - rowSize / 2) * spacing,
            fz: (col - rowSize / 2) * spacing,
            activation: 0,
        })
    }

    const hiddenSpacing = 6;
    for (let i = 0; i < HIDDEN_COUNT; i += 1) {
        nodes.push({
            id: `hidden-${i}`,
            layer: 'hidden',
            fx: 50,
            fy: (i - HIDDEN_COUNT / 2) * hiddenSpacing,
            fz: 0,
            activation: 0,
        })
    }

    const outputSpacing = 20;
    for (let i = 0; i < OUTPUT_COUNT; i += 1) {
        nodes.push({
            id: `output-${i}`,
            layer: 'output',
            fx: 200,
            fy: (i - OUTPUT_COUNT / 2) * outputSpacing,
            fz: 0,
            activation: 0,
            error: 0,
        })
    }
    return nodes
}

const initialGraph = {
    nodes: buildNodes(),
    links: [] as GraphLink[],
}

export default function TrainImages() {
    const [mode, setMode] = useState<'fast' | 'slow'>('fast')
    const [epochs, setEpochs] = useState(10)
    const [useTrained, setUseTrained] = useState(false)
    const [status, setStatus] = useState({
        epoch: 0,
        iteration: 0,
        loss: 0,
        accuracy: 0,
    })
    const [infoMessages, setInfoMessages] = useState<string[]>([])
    const [graphData, setGraphData] = useState(initialGraph)
    const [lossPoints, setLossPoints] = useState<number[]>([])
    const [accuracyPoints, setAccuracyPoints] = useState<number[]>([])
    const [iterations, setIterations] = useState<number[]>([])
    const [accuracyIterations, setAccuracyIterations] = useState<number[]>([])
    const socketRef = useRef<WebSocket | null>(null)
    const graphRef = useRef<any>(null)
    const iterationRef = useRef(0)

    useEffect(() => {
        if (graphRef.current) {
            const controls = graphRef.current.controls();

            // controls.target.set(50, 0, 0);

            // Smooth navigation
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;

            // Optional: Restrict zoom range so they don't get "lost" in space
            controls.minDistance = 100;
            controls.maxDistance = 700;

            // Adjust rotation speed (lower is usually more "heavy" and premium)
            controls.rotateSpeed = 0.7;
        }
    }, []);

    useEffect(() => {
        const socket = new WebSocket(`${WS_BASE_URL}/ws/train`)
        socketRef.current = socket

        socket.onmessage = (event) => {
            const message = JSON.parse(event.data) as ServerMessage
            if (message.type === 'weights') {

                // 1. Calculate Hidden Activations (size should be 64)
                // We need to know how much each HIDDEN neuron is receiving
                const hiddenCount = message.hidden[0].length; // The 64
                const hiddenActivation = new Array(hiddenCount).fill(0);

                message.hidden.forEach((row) => {
                    row.forEach((weight, j) => {
                        hiddenActivation[j] += Math.abs(weight);
                    });
                });
                // Average it out
                const finalHiddenActivation = hiddenActivation.map(v => v / message.hidden.length);

                // 2. Calculate Output Activations (size should be 10)
                const outputCount = message.output[0].length; // The 10
                const outputActivation = new Array(outputCount).fill(0);

                message.output.forEach((row) => {
                    row.forEach((weight, j) => {
                        outputActivation[j] += Math.abs(weight);
                    });
                });
                const finalOutputActivation = outputActivation.map(v => v / message.output.length);

                // 3. Build Links with a STRICT threshold to keep performance high
                const links: GraphLink[] = [];
                const LINK_THRESHOLD = 0.1;

                message.hidden.forEach((row, i) => {
                    row.forEach((weight, j) => {
                        if (Math.abs(weight) > LINK_THRESHOLD) {
                            links.push({source: `input-${j}`, target: `hidden-${i}`, weight});
                        }
                    });
                });

                message.output.forEach((row, i) => {
                    row.forEach((weight, j) => {
                        if (Math.abs(weight) > (LINK_THRESHOLD * 0.6)) {
                            links.push({source: `hidden-${j}`, target: `output-${i}`, weight});
                        }
                    });
                });

                setGraphData((prev) => ({
                    nodes: prev.nodes.map((node) => {
                        const idParts = node.id.split('-');
                        const index = parseInt(idParts[1], 10);

                        if (node.layer === 'hidden') {
                            return {
                                ...node,
                                // Use fallback || 0.1 so nodes never have 0 size (disappearing)
                                activation: finalHiddenActivation[index] || 0.1,
                            };
                        }
                        if (node.layer === 'output') {
                            return {
                                ...node,
                                activation: finalOutputActivation[index] || 0.1,
                                error: (finalOutputActivation[index] || 0) - status.accuracy,
                            };
                        }
                        return node;
                    }),
                    links,
                }));
            }
            if (message.type === 'epoch') {
                setStatus((prev) => ({...prev, epoch: message.value}))
            }
            if (message.type === 'iteration') {
                iterationRef.current = message.value
                setStatus((prev) => ({...prev, iteration: message.value}))
            }
            if (message.type === 'loss') {
                setStatus((prev) => ({...prev, loss: message.value}))
                setIterations((prev) => [...prev, message.x])
                setLossPoints((prev) => [...prev, message.value])
            }
            if (message.type === 'accuracy') {
                setStatus((prev) => ({...prev, accuracy: message.value}))
                setAccuracyIterations((prev) => [...prev, iterationRef.current])
                setAccuracyPoints((prev) => [...prev, message.value])
            }
            if (message.type === 'info') {
                setInfoMessages((prev) => [message.message, ...prev].slice(0, 6))
            }
        }

        return () => {
            socket.close()
        }
    }, [])

    useEffect(() => {
        if (!graphRef.current) return
        graphRef.current.cameraPosition({z: 260, x: 0, y: 0})
    }, [])

    const handleSend = (payload: ClientMessage) => {
        if (socketRef.current?.readyState === WebSocket.OPEN) {
            socketRef.current.send(JSON.stringify(payload))
        }
    }

    const handleStart = () => {
        setLossPoints([])
        setAccuracyPoints([])
        setIterations([])
        setAccuracyIterations([])
        handleSend({
            type: 'start',
            mode,
            epochs,
            learningRate: 0.1,
            batchSize: 1,
            use_trained: useTrained,
        })
    }

    const lossData = useMemo(
        () => ({
            labels: iterations,
            datasets: [
                {
                    label: 'Error',
                    data: lossPoints,
                    borderColor: '#ff6b35',
                    backgroundColor: 'rgba(255, 107, 53, 0.2)',
                    tension: 0.2,
                },
            ],
        }),
        [iterations, lossPoints],
    )

    const accuracyData = useMemo(
        () => ({
            labels: accuracyIterations,
            datasets: [
                {
                    label: 'Exactitud',
                    data: accuracyPoints,
                    borderColor: '#2d7dd2',
                    backgroundColor: 'rgba(45, 125, 210, 0.2)',
                    tension: 0.2,
                },
            ],
        }),
        [accuracyIterations, accuracyPoints],
    )

    const chartOptions = useMemo(
        () => ({
            responsive: true,
            plugins: {
                legend: {
                    display: false,
                },
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Iteracion',
                    },
                },
                y: {
                    title: {
                        display: true,
                        text: 'Valor',
                    },
                },
            },
        }),
        [],
    )

    return (
        <main className="page">
            <header className="page-header">
                <div>
                    <h1 className="title">Entrenamiento con imagenes</h1>
                </div>
                <Link to="/" className="nav-link">
                    Volver al menu
                </Link>
            </header>
            <section className="stack">
                <div className="card stack">
                    <div className="pill-group">
                        <button type="button" className="cta" onClick={handleStart}>
                            Reproducir
                        </button>
                        <button
                            type="button"
                            className="cta secondary"
                            onClick={() => handleSend({type: 'stop'})}
                        >
                            Detener
                        </button>
                        <button
                            type="button"
                            className="cta secondary"
                            onClick={() => handleSend({type: 'resume'})}
                        >
                            Reanudar
                        </button>
                    </div>
                    <div className="grid-2">
                        <div className="field">
                            <label>Modo</label>
                            <div className="pill-group">
                                <button
                                    type="button"
                                    className={`pill ${mode === 'fast' ? 'active' : ''}`}
                                    onClick={() => setMode('fast')}
                                >
                                    Modo rapido
                                </button>
                                <button
                                    type="button"
                                    className={`pill ${mode === 'slow' ? 'active' : ''}`}
                                    onClick={() => setMode('slow')}
                                >
                                    Modo lento
                                </button>
                            </div>
                        </div>
                        <div className="field">
                            <label>Epocas</label>
                            <input
                                type="number"
                                min={1}
                                value={epochs}
                                onChange={(event) => setEpochs(Number(event.target.value))}
                            />
                        </div>
                        <div className="field">
                            <label>Usar modelo entrenado</label>
                            <select
                                value={useTrained ? 'yes' : 'no'}
                                onChange={(event) => setUseTrained(event.target.value === 'yes')}
                            >
                                <option value="no">No</option>
                                <option value="yes">Si</option>
                            </select>
                        </div>
                    </div>
                </div>

                <div className="card stack">
                    <h2>Estado</h2>
                    <div className="status-grid">
                        <div className="status-item">
                            <span>Epoca actual</span>
                            <strong>{status.epoch}</strong>
                        </div>
                        <div className="status-item">
                            <span>Iteracion actual</span>
                            <strong>{status.iteration}</strong>
                        </div>
                        <div className="status-item">
                            <span>Error actual</span>
                            <strong>{status.loss.toFixed(4)}</strong>
                        </div>
                        <div className="status-item">
                            <span>Exactitud actual</span>
                            <strong>{status.accuracy.toFixed(4)}</strong>
                        </div>
                    </div>
                    <div>
                        <h3>Mensajes</h3>
                        <div className="log">
                            {infoMessages.length > 0
                                ? infoMessages.map((message, index) => (
                                    <div key={`${message}-${index}`}>{message}</div>
                                ))
                                : 'Sin mensajes por ahora.'}
                        </div>
                    </div>
                </div>

                <div className="card stack">
                    <h2>Visualizacion ANN</h2>
                    <div className="graph-wrapper">
                        <ForceGraph3D
                            ref={graphRef}
                            controlType="orbit"
                            graphData={graphData}

                            backgroundColor="#0c0f12"
                            showNavInfo={false}

                            nodeVal={(node: any) => Math.max(1, (node.activation || 0) * 10)}
                            onNodeClick={(node: any) => {

                                const distance = 60;
                                const distRatio = 1 + distance / Math.hypot(node.x, node.y, node.z);

                                graphRef.current.cameraPosition(
                                    {x: node.x * distRatio, y: node.y * distRatio, z: node.z * distRatio},
                                    node,
                                    300
                                );
                            }}
                            nodeRelSize={3}
                            nodeColor={(node: any) => {
                                if (node.layer === 'output') return node.activation > 0.5 ? '#4CAF50' : '#F44336';
                                if (node.layer === 'hidden') return '#FFC107';
                                return `rgba(200, 200, 200, ${0.2 + node.activation * 0.8})`;
                            }}

                            // Link hover label
                            linkLabel={(link: any) => `
    <div style="background: #222; color: #fff; padding: 5px; border-radius: 4px; border: 1px solid #555;">
      <b>${link.source.id} - ${link.target.id}</b><br/>
      Peso: <span style="color: ${link.weight > 0 ? '#4CAF50' : '#F44336'}">
        ${link.weight.toFixed(4)}
      </span>
    </div>
  `}

                            nodeLabel={(node: any) => `
    <div style="background: #222; color: #fff; padding: 5px; border-radius: 4px; border: 1px solid #555;">
      <b>Nodo: ${node.layer.toUpperCase()}</b><br/>
      ID: ${node.id}<br/>
      Activación: ${node.activation.toFixed(4)}
    </div>
  `}

                            enablePointerInteraction={true}

                            linkWidth={(link: any) => Math.abs(link.weight)}
                            linkColor={(link: any) => link.weight > 0 ? 'rgba(255,107,53,0.3)' : 'rgba(45,125,210,0.3)'}

                            linkDirectionalParticles={1}
                            linkDirectionalParticleSpeed={0.01}

                            enableNavigationControls={true} // Allow the user to rotate the 3D grid!
                            enableNodeDrag={false}
                        />
                    </div>
                </div>

                <div className="grid-2">
                    <div className="chart-card">
                        <h3>Error vs Iteracion</h3>
                        <Line data={lossData} options={chartOptions}/>
                    </div>
                    <div className="chart-card">
                        <h3>Exactitud vs Iteracion</h3>
                        <Line data={accuracyData} options={chartOptions}/>
                    </div>
                </div>
            </section>
        </main>
    )
}
