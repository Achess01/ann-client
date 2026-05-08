import { useEffect, useMemo, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
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

const INPUT_COUNT = 784
const HIDDEN_COUNT = 64
const OUTPUT_COUNT = 10

const SLOW_REDRAW_EVERY = 2
const WEIGHT_MIN_WIDTH = 0.2
const WEIGHT_MAX_WIDTH = 1.4
const INPUT_RADIUS = 1.6
const HIDDEN_RADIUS = 3.4
const OUTPUT_RADIUS = 5.2
const CANVAS_MARGIN = 36
const INPUT_VISIBLE_START = 16
const INPUT_VISIBLE_END = 16
const HIDDEN_VISIBLE_START = 8
const HIDDEN_VISIBLE_END = 8

type LayerLayout = {
  x: number
  ys: number[]
}

type Layout = {
  input: LayerLayout
  hidden: LayerLayout
  output: LayerLayout
}

const buildLayerYs = (count: number, height: number) => {
  if (count === 1) return [height / 2]
  const spacing = height / (count - 1)
  return Array.from({ length: count }, (_, i) => i * spacing)
}

const buildLayout = (width: number, height: number): Layout => {
  const availableHeight = Math.max(100, height - CANVAS_MARGIN * 2)
  const startY = CANVAS_MARGIN
  const inputYs = buildLayerYs(INPUT_COUNT, availableHeight).map(
    (y) => y + startY,
  )
  const hiddenYs = buildLayerYs(HIDDEN_COUNT, availableHeight).map(
    (y) => y + startY,
  )
  const outputYs = buildLayerYs(OUTPUT_COUNT, availableHeight).map(
    (y) => y + startY,
  )
  return {
    input: { x: width * 0.18, ys: inputYs },
    hidden: { x: width * 0.5, ys: hiddenYs },
    output: { x: width * 0.82, ys: outputYs },
  }
}

const buildVisibleIndices = (count: number, start: number, end: number) => {
  const safeStart = Math.min(start, count)
  const safeEnd = Math.min(end, Math.max(0, count - safeStart))
  const indices: number[] = []
  for (let i = 0; i < safeStart; i += 1) {
    indices.push(i)
  }
  for (let i = count - safeEnd; i < count; i += 1) {
    if (i >= 0) indices.push(i)
  }
  return indices
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
  const [canvasSize, setCanvasSize] = useState({ width: 720, height: 560 })
  const [lossPoints, setLossPoints] = useState<number[]>([])
  const [accuracyPoints, setAccuracyPoints] = useState<number[]>([])
  const [iterations, setIterations] = useState<number[]>([])
  const [accuracyIterations, setAccuracyIterations] = useState<number[]>([])
  const socketRef = useRef<WebSocket | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const containerRef = useRef<HTMLDivElement | null>(null)
  const iterationRef = useRef(0)
  const weightsHiddenRef = useRef<number[][] | null>(null)
  const weightsOutputRef = useRef<number[][] | null>(null)
  const maxAbsWeightRef = useRef(1)
  const activationHiddenRef = useRef<number[]>(new Array(HIDDEN_COUNT).fill(0))
  const activationOutputRef = useRef<number[]>(new Array(OUTPUT_COUNT).fill(0))
  const redrawPendingRef = useRef(false)

  useEffect(() => {
    const socket = new WebSocket(`${WS_BASE_URL}/ws/train`)
    socketRef.current = socket

    socket.onmessage = (event) => {
      const message = JSON.parse(event.data) as ServerMessage
      if (message.type === 'weights') {
        let maxAbsWeight = 0
        message.hidden.forEach((row) => {
          row.forEach((weight) => {
            maxAbsWeight = Math.max(maxAbsWeight, Math.abs(weight))
          })
        })
        message.output.forEach((row) => {
          row.forEach((weight) => {
            maxAbsWeight = Math.max(maxAbsWeight, Math.abs(weight))
          })
        })
        weightsHiddenRef.current = message.hidden
        weightsOutputRef.current = message.output
        maxAbsWeightRef.current = maxAbsWeight || 1
        activationHiddenRef.current = message.hidden.map((row) =>
          row.reduce((sum, value) => sum + Math.abs(value), 0) / row.length,
        )
        activationOutputRef.current = message.output.map((row) =>
          row.reduce((sum, value) => sum + Math.abs(value), 0) / row.length,
        )
        if (mode === 'slow' && iterationRef.current % SLOW_REDRAW_EVERY !== 0) {
          return
        }
        scheduleDraw()
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
    if (!containerRef.current) return
    const observer = new ResizeObserver((entries) => {
      const entry = entries[0]
      if (!entry) return
      const { width, height } = entry.contentRect
      setCanvasSize({ width, height })
    })
    observer.observe(containerRef.current)
    return () => observer.disconnect()
  }, [])

  useEffect(() => {
    scheduleDraw()
  }, [canvasSize])

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

  const scheduleDraw = () => {
    if (redrawPendingRef.current) return
    redrawPendingRef.current = true
    window.requestAnimationFrame(() => {
      redrawPendingRef.current = false
      drawCanvas()
    })
  }

  const drawCanvas = () => {
    const canvas = canvasRef.current
    if (!canvas) return
    const context = canvas.getContext('2d')
    if (!context) return
    const { width, height } = canvasSize
    canvas.width = Math.max(1, Math.floor(width * window.devicePixelRatio))
    canvas.height = Math.max(1, Math.floor(height * window.devicePixelRatio))
    context.setTransform(window.devicePixelRatio, 0, 0, window.devicePixelRatio, 0, 0)
    context.clearRect(0, 0, width, height)

    const layout = buildLayout(width, height)
    const visibleInput = buildVisibleIndices(
      INPUT_COUNT,
      INPUT_VISIBLE_START,
      INPUT_VISIBLE_END,
    )
    const visibleHidden = buildVisibleIndices(
      HIDDEN_COUNT,
      HIDDEN_VISIBLE_START,
      HIDDEN_VISIBLE_END,
    )
    const visibleOutput = buildVisibleIndices(OUTPUT_COUNT, OUTPUT_COUNT, 0)
    const visibleInputSet = new Set(visibleInput)
    const visibleHiddenSet = new Set(visibleHidden)

    context.fillStyle = 'rgba(255,255,255,0.6)'
    context.font = '12px "IBM Plex Mono", ui-monospace'
    const legendY = 18
    context.fillText(
      `Entrada: I0-I${INPUT_VISIBLE_START - 1}, I${
        INPUT_COUNT - INPUT_VISIBLE_END
      }-I${INPUT_COUNT - 1}`,
      16,
      legendY,
    )
    context.fillText(
      `Oculta: H0-H${HIDDEN_VISIBLE_START - 1}, H${
        HIDDEN_COUNT - HIDDEN_VISIBLE_END
      }-H${HIDDEN_COUNT - 1}`,
      16,
      legendY + 16,
    )
    context.fillText('Salida: O0-O9', 16, legendY + 32)
    const hiddenWeights = weightsHiddenRef.current
    const outputWeights = weightsOutputRef.current
    const maxAbsWeight = maxAbsWeightRef.current || 1

    if (hiddenWeights && outputWeights) {
      for (let i = 0; i < INPUT_COUNT; i += 1) {
        if (!visibleInputSet.has(i)) continue
        const x1 = layout.input.x
        const y1 = layout.input.ys[i]
        const row = hiddenWeights[i]
        for (let j = 0; j < HIDDEN_COUNT; j += 1) {
          if (!visibleHiddenSet.has(j)) continue
          const weight = row[j] ?? 0
          const normalized = Math.abs(weight) / maxAbsWeight
          const widthValue = WEIGHT_MIN_WIDTH + normalized * WEIGHT_MAX_WIDTH
          const alpha = 0.08 + normalized * 0.25
          context.strokeStyle =
            weight >= 0
              ? `rgba(255, 107, 53, ${alpha})`
              : `rgba(45, 125, 210, ${alpha})`
          context.lineWidth = widthValue
          context.beginPath()
          context.moveTo(x1, y1)
          context.lineTo(layout.hidden.x, layout.hidden.ys[j])
          context.stroke()
        }
      }

      for (let i = 0; i < HIDDEN_COUNT; i += 1) {
        if (!visibleHiddenSet.has(i)) continue
        const x1 = layout.hidden.x
        const y1 = layout.hidden.ys[i]
        const row = outputWeights[i]
        for (let j = 0; j < OUTPUT_COUNT; j += 1) {
          const weight = row[j] ?? 0
          const normalized = Math.abs(weight) / maxAbsWeight
          const widthValue = WEIGHT_MIN_WIDTH + normalized * WEIGHT_MAX_WIDTH
          const alpha = 0.12 + normalized * 0.35
          context.strokeStyle =
            weight >= 0
              ? `rgba(255, 107, 53, ${alpha})`
              : `rgba(45, 125, 210, ${alpha})`
          context.lineWidth = widthValue
          context.beginPath()
          context.moveTo(x1, y1)
          context.lineTo(layout.output.x, layout.output.ys[j])
          context.stroke()
        }
      }
    }

    const hiddenActivation = activationHiddenRef.current
    const outputActivation = activationOutputRef.current
    const activationMax = Math.max(
      0.0001,
      ...hiddenActivation,
      ...outputActivation,
    )

    for (let i = 0; i < INPUT_COUNT; i += 1) {
      if (!visibleInputSet.has(i)) continue
      context.fillStyle = 'rgba(245, 245, 245, 0.85)'
      context.beginPath()
      context.arc(layout.input.x, layout.input.ys[i], INPUT_RADIUS, 0, Math.PI * 2)
      context.fill()
      context.fillStyle = 'rgba(255,255,255,0.7)'
      context.font = '10px "IBM Plex Mono", ui-monospace'
      context.fillText(`I${i}`, layout.input.x - 26, layout.input.ys[i] + 3)
    }

    for (let i = 0; i < HIDDEN_COUNT; i += 1) {
      if (!visibleHiddenSet.has(i)) continue
      const activation = hiddenActivation[i] || 0
      const intensity = Math.min(1, activation / activationMax)
      const radius = HIDDEN_RADIUS + intensity * 2.2
      context.fillStyle = `rgba(255, 200, 120, ${0.5 + intensity * 0.4})`
      context.beginPath()
      context.arc(layout.hidden.x, layout.hidden.ys[i], radius, 0, Math.PI * 2)
      context.fill()
      context.fillStyle = 'rgba(255,255,255,0.75)'
      context.font = '10px "IBM Plex Mono", ui-monospace'
      context.fillText(`H${i}`, layout.hidden.x + 8, layout.hidden.ys[i] + 3)
    }

    for (let i = 0; i < OUTPUT_COUNT; i += 1) {
      const activation = outputActivation[i] || 0
      const intensity = Math.min(1, activation / activationMax)
      const radius = OUTPUT_RADIUS + intensity * 2.5
      context.fillStyle = `rgba(120, 200, 255, ${0.6 + intensity * 0.35})`
      context.beginPath()
      context.arc(layout.output.x, layout.output.ys[i], radius, 0, Math.PI * 2)
      context.fill()
      context.strokeStyle = 'rgba(255,255,255,0.25)'
      context.lineWidth = 1
      context.stroke()
      context.fillStyle = 'rgba(255,255,255,0.85)'
      context.font = '10px "IBM Plex Mono", ui-monospace'
      context.fillText(`O${i}`, layout.output.x + 10, layout.output.ys[i] + 3)
    }
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
            <div className="graph-wrapper" ref={containerRef}>
              <canvas ref={canvasRef} aria-label="Visualizacion ANN"></canvas>
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
