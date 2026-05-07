import { useState } from 'react'
import { Link } from 'react-router-dom'
import { API_BASE_URL } from '../config'

type TrainResponse = {
  model_path: string
  metrics: {
    final_train_loss: number
    final_train_accuracy: number
    test_loss: number
    test_accuracy: number
  }
}

export default function TrainMnist() {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<TrainResponse | null>(null)

  const handleTrain = async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await fetch(`${API_BASE_URL}/train/mnist`, {
        method: 'POST',
      })
      if (!response.ok) {
        throw new Error('No se pudo iniciar el entrenamiento')
      }
      const data = (await response.json()) as TrainResponse
      setResult(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Error inesperado')
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="page">
      <header className="page-header">
        <div>
          <h1 className="title">Entrenamiento con MNIST</h1>
        </div>
        <Link to="/" className="nav-link">
          Volver al menu
        </Link>
      </header>
      <section className="stack">
        <div className="card stack">
          <button type="button" className="cta" onClick={handleTrain}>
            {loading ? 'Entrenando...' : 'Iniciar entrenamiento'}
          </button>
          {error && <p className="hint">{error}</p>}
        </div>
        <div className="card stack">
          <h2>Resultados</h2>
          {result ? (
            <div className="status-grid">
              <div className="status-item">
                <span>Error final (entrenamiento)</span>
                <strong>{result.metrics.final_train_loss.toFixed(4)}</strong>
              </div>
              <div className="status-item">
                <span>Exactitud final (entrenamiento)</span>
                <strong>{result.metrics.final_train_accuracy.toFixed(4)}</strong>
              </div>
              <div className="status-item">
                <span>Error (test)</span>
                <strong>{result.metrics.test_loss.toFixed(4)}</strong>
              </div>
              <div className="status-item">
                <span>Exactitud (test)</span>
                <strong>{result.metrics.test_accuracy.toFixed(4)}</strong>
              </div>
            </div>
          ) : (
            <p className="hint">Sin resultados por ahora.</p>
          )}
        </div>
      </section>
    </main>
  )
}
