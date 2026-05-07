import { useEffect, useMemo, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
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
import { Line } from 'react-chartjs-2'
import { WS_BASE_URL } from '../config'
import type { ClientMessage, ServerMessage } from '../types'

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
  x: number
  y: number
  z: number
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
  for (let i = 0; i < INPUT_COUNT; i += 1) {
    nodes.push({
      id: `input-${i}`,
      layer: 'input',
      x: -120,
      y: (i - INPUT_COUNT / 2) * 0.6,
      z: 0,
      activation: 0,
    })
  }
  for (let i = 0; i < HIDDEN_COUNT; i += 1) {
    nodes.push({
      id: `hidden-${i}`,
      layer: 'hidden',
      x: 0,
      y: (i - HIDDEN_COUNT / 2) * 7,
      z: 0,
      activation: 0,
    })
  }
  for (let i = 0; i < OUTPUT_COUNT; i += 1) {
    nodes.push({
      id: `output-${i}`,
      layer: 'output',
      x: 120,
      y: (i - OUTPUT_COUNT / 2) * 24,
      z: 0,
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
    const socket = new WebSocket(`${WS_BASE_URL}/ws/train`)
    socketRef.current = socket

    socket.onmessage = (event) => {
      const message = JSON.parse(event.data) as ServerMessage
      if (message.type === 'weights') {
        const hiddenActivation = message.hidden.map((row) =>
          row.reduce((sum, value) => sum + Math.abs(value), 0) / row.length,
        )
        const outputActivation = message.output.map((row) =>
          row.reduce((sum, value) => sum + Math.abs(value), 0) / row.length,
        )
        const links: GraphLink[] = []
        message.hidden.forEach((row, i) => {
          row.forEach((weight, j) => {
            links.push({
              source: `input-${i}`,
              target: `hidden-${j}`,
              weight,
            })
          })
        })
        message.output.forEach((row, i) => {
          row.forEach((weight, j) => {
            links.push({
              source: `hidden-${i}`,
              target: `output-${j}`,
              weight,
            })
          })
        })
        setGraphData((prev) => ({
          nodes: prev.nodes.map((node) => {
            if (node.layer === 'hidden') {
              const index = Number(node.id.replace('hidden-', ''))
              return {
                ...node,
                activation: hiddenActivation[index] ?? node.activation,
              }
            }
            if (node.layer === 'output') {
              const index = Number(node.id.replace('output-', ''))
              const activation = outputActivation[index] ?? node.activation
              return {
                ...node,
                activation,
                error: activation - status.accuracy,
              }
            }
            return node
          }),
          links,
        }))
      }
      if (message.type === 'epoch') {
        setStatus((prev) => ({ ...prev, epoch: message.value }))
      }
      if (message.type === 'iteration') {
        iterationRef.current = message.value
        setStatus((prev) => ({ ...prev, iteration: message.value }))
      }
      if (message.type === 'loss') {
        setStatus((prev) => ({ ...prev, loss: message.value }))
        setIterations((prev) => [...prev, message.x])
        setLossPoints((prev) => [...prev, message.value])
      }
      if (message.type === 'accuracy') {
        setStatus((prev) => ({ ...prev, accuracy: message.value }))
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
    graphRef.current.cameraPosition({ z: 260, x: 0, y: 0 })
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
              onClick={() => handleSend({ type: 'stop' })}
            >
              Detener
            </button>
            <button
              type="button"
              className="cta secondary"
              onClick={() => handleSend({ type: 'resume' })}
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
              graphData={graphData}
              nodeAutoColorBy="layer"
              linkColor={(link) =>
                link.weight >= 0 ? 'rgba(255,107,53,0.7)' : 'rgba(45,125,210,0.7)'
              }
              linkWidth={(link) => Math.min(4, Math.abs(link.weight) * 4 + 0.2)}
              nodeVal={(node) => 1 + Math.abs(node.activation) * 4}
              nodeColor={(node) => {
                if (node.layer === 'output') {
                  const intensity = Math.min(1, Math.abs(node.error ?? 0))
                  return node.error && node.error > 0
                    ? `rgba(255,80,80,${0.5 + intensity * 0.5})`
                    : `rgba(80,160,255,${0.5 + intensity * 0.5})`
                }
                return node.layer === 'hidden'
                  ? 'rgba(255, 200, 120, 0.9)'
                  : 'rgba(240, 240, 240, 0.85)'
              }}
              backgroundColor="#0c0f12"
              enableNavigationControls={false}
              enableZoomInteraction={false}
              enablePanInteraction={false}
              enablePointerInteraction={false}
              enableNodeDrag={false}
              showNavInfo={false}
            />
          </div>
        </div>

        <div className="grid-2">
          <div className="chart-card">
            <h3>Error vs Iteracion</h3>
            <Line data={lossData} options={chartOptions} />
          </div>
          <div className="chart-card">
            <h3>Exactitud vs Iteracion</h3>
            <Line data={accuracyData} options={chartOptions} />
          </div>
        </div>
      </section>
    </main>
  )
}
