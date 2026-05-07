import { Link } from 'react-router-dom'

export default function MainMenu() {
  return (
    <main className="page">
      <header className="page-header">
        <h1 className="title">Menu principal</h1>
      </header>
      <section className="card stack">
        <p className="hint">
          Selecciona un flujo de entrenamiento para comenzar.
        </p>
        <div className="grid-2">
          <Link to="/train-mnist" className="cta">
            Entrenar con MNIST
          </Link>
          <Link to="/train-images" className="cta secondary">
            Entrenar con imagenes
          </Link>
        </div>
      </section>
    </main>
  )
}
