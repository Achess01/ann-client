import {type ChangeEvent, useEffect, useMemo, useRef, useState} from 'react'
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

type TrainingImage = {
    id: string
    src: string
    label: number
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

const createImageId = () => `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`

const readFileAsDataUrl = (file: File) =>
    new Promise<string>((resolve, reject) => {
        const reader = new FileReader()
        reader.onload = () => {
            if (typeof reader.result === 'string') {
                resolve(reader.result)
                return
            }
            reject(new Error('No se pudo leer la imagen.'))
        }
        reader.onerror = () => reject(new Error('No se pudo leer la imagen.'))
        reader.readAsDataURL(file)
    })

const loadImage = (src: string) =>
    new Promise<HTMLImageElement>((resolve, reject) => {
        const image = new Image()
        image.onload = () => resolve(image)
        image.onerror = () => reject(new Error('No se pudo procesar la imagen.'))
        image.src = src
    })

const imageToGrayscaleBase64 = async (src: string) => {
    const image = await loadImage(src)
    const canvas = document.createElement('canvas')
    canvas.width = 28
    canvas.height = 28

    const context = canvas.getContext('2d')
    if (!context) {
        throw new Error('No se pudo preparar el lienzo de imagen.')
    }

    const size = Math.min(image.width, image.height)
    const offsetX = Math.floor((image.width - size) / 2)
    const offsetY = Math.floor((image.height - size) / 2)

    context.drawImage(image, offsetX, offsetY, size, size, 0, 0, 28, 28)

    const imageData = context.getImageData(0, 0, 28, 28)
    const {data} = imageData

    for (let i = 0; i < data.length; i += 4) {
        const gray = Math.round(data[i] * 0.299 + data[i + 1] * 0.587 + data[i + 2] * 0.114)
        data[i] = gray
        data[i + 1] = gray
        data[i + 2] = gray
        data[i + 3] = 255
    }

    context.putImageData(imageData, 0, 0)

    return canvas.toDataURL('image/png').split(',')[1]
}

export default function TrainImages() {
    const [mode, setMode] = useState<'fast' | 'slow'>('fast')
    const [epochs, setEpochs] = useState(10)
    const [useTrained, setUseTrained] = useState(false)
    const [trainingImages, setTrainingImages] = useState<TrainingImage[]>([])
    const [cameraActive, setCameraActive] = useState(false)
    const [cameraError, setCameraError] = useState<string | null>(null)
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
    const fileInputRef = useRef<HTMLInputElement | null>(null)
    const videoRef = useRef<HTMLVideoElement | null>(null)
    const cameraStreamRef = useRef<MediaStream | null>(null)
    const cameraCanvasRef = useRef<HTMLCanvasElement | null>(null)

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

    useEffect(() => {
        return () => {
            cameraStreamRef.current?.getTracks().forEach((track) => track.stop())
        }
    }, [])

    const handleSend = async (payload: ClientMessage) => {
        if (socketRef.current?.readyState === WebSocket.OPEN) {
            if (payload.type === 'start') {
                const images_b64 = await Promise.all(
                    trainingImages.map((image) => imageToGrayscaleBase64(image.src)),
                )
                const labels = trainingImages.map((image) => image.label)

                socketRef.current.send(
                    JSON.stringify({
                        ...payload,
                        images_b64,
                        labels,
                        num_images: images_b64.length,
                    }),
                )
                return
            }

            socketRef.current.send(JSON.stringify(payload))
        }
    }

    const handleFilesSelected = async (event: ChangeEvent<HTMLInputElement>) => {
        const files = Array.from(event.target.files ?? [])
        if (files.length === 0) return

        const nextImages = await Promise.all(
            files.map(async (file) => ({
                id: createImageId(),
                src: await readFileAsDataUrl(file),
                label: 0,
            })),
        )

        setTrainingImages((prev) => [...prev, ...nextImages])
        event.target.value = ''
    }

    const handleLabelChange = (id: string, label: number) => {
        setTrainingImages((prev) =>
            prev.map((image) => (image.id === id ? {...image, label} : image)),
        )
    }

    const handleRemoveImage = (id: string) => {
        setTrainingImages((prev) => prev.filter((image) => image.id !== id))
    }

    const startCamera = async () => {
        setCameraError(null)

        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: true,
                audio: false,
            })

            cameraStreamRef.current = stream
            setCameraActive(true)

            if (videoRef.current) {
                videoRef.current.srcObject = stream
                await videoRef.current.play()
            }
        } catch (error) {
            setCameraError(error instanceof Error ? error.message : 'No se pudo abrir la camara.')
        }
    }

    const stopCamera = () => {
        cameraStreamRef.current?.getTracks().forEach((track) => track.stop())
        cameraStreamRef.current = null

        if (videoRef.current) {
            videoRef.current.srcObject = null
        }

        setCameraActive(false)
    }

    const captureCameraImage = () => {
        const video = videoRef.current
        const canvas = cameraCanvasRef.current

        if (!video || !canvas || video.videoWidth === 0 || video.videoHeight === 0) {
            return
        }

        canvas.width = video.videoWidth
        canvas.height = video.videoHeight

        const context = canvas.getContext('2d')
        if (!context) return

        context.drawImage(video, 0, 0, canvas.width, canvas.height)

        setTrainingImages((prev) => [
            ...prev,
            {
                id: createImageId(),
                src: canvas.toDataURL('image/png'),
                label: 0,
            },
        ])
    }

    const clearImages = () => {
        setTrainingImages([])
    }

    const handleStart = () => {
        setLossPoints([])
        setAccuracyPoints([])
        setIterations([])
        setAccuracyIterations([])
        void handleSend({
            type: 'start',
            mode,
            epochs,
            learningRate: 0.1,
            batchSize: 1,
            use_trained: useTrained,
            images_b64: [],
            labels: [],
            num_images: 0,
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
                            onClick={() => void handleSend({type: 'stop'})}
                        >
                            Detener
                        </button>
                        <button
                            type="button"
                            className="cta secondary"
                            onClick={() => void handleSend({type: 'resume'})}
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
                    <h2>Imagenes de entrenamiento</h2>
                    <div className="pill-group">
                        <button type="button" className="cta" onClick={() => fileInputRef.current?.click()}>
                            Cargar archivos
                        </button>
                        {!cameraActive ? (
                            <button type="button" className="cta secondary" onClick={() => void startCamera()}>
                                Activar camara
                            </button>
                        ) : (
                            <>
                                <button type="button" className="cta secondary" onClick={captureCameraImage}>
                                    Capturar imagen
                                </button>
                                <button type="button" className="cta secondary" onClick={stopCamera}>
                                    Apagar camara
                                </button>
                            </>
                        )}
                        <button type="button" className="cta secondary" onClick={clearImages}>
                            Limpiar lista
                        </button>
                    </div>

                    <input
                        ref={fileInputRef}
                        type="file"
                        accept="image/*"
                        multiple
                        hidden
                        onChange={handleFilesSelected}
                    />

                    {cameraError ? <div className="hint">{cameraError}</div> : null}

                    <div className="camera-panel">
                        <video ref={videoRef} className="camera-preview" autoPlay playsInline muted />
                        <canvas ref={cameraCanvasRef} className="sr-only" />
                    </div>

                    {trainingImages.length > 0 ? (
                        <div className="image-grid">
                            {trainingImages.map((image) => (
                                <div key={image.id} className="image-item">
                                    <img src={image.src} alt={`Training ${image.id}`} className="image-preview" />
                                    <div className="field">
                                        <label>Digito</label>
                                        <select
                                            value={image.label}
                                            onChange={(event) => handleLabelChange(image.id, Number(event.target.value))}
                                        >
                                            {Array.from({length: 10}, (_, digit) => (
                                                <option key={digit} value={digit}>
                                                    {digit}
                                                </option>
                                            ))}
                                        </select>
                                    </div>
                                    <button type="button" className="cta secondary" onClick={() => handleRemoveImage(image.id)}>
                                        Eliminar
                                    </button>
                                </div>
                            ))}
                        </div>
                    ) : (
                        <div className="hint">Aun no hay imagenes cargadas.</div>
                    )}
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
