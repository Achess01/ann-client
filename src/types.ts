export type ClientMessage =
  | {
      type: 'start'
      mode: 'fast' | 'slow'
      epochs: number
      learningRate: number
      batchSize: number
      use_trained: boolean
    }
  | { type: 'stop' }
  | { type: 'resume' }

export type ServerMessage =
  | { type: 'weights'; hidden: number[][]; output: number[][] }
  | { type: 'epoch'; value: number }
  | { type: 'iteration'; value: number }
  | { type: 'loss'; x: number; value: number }
  | { type: 'accuracy'; value: number }
  | { type: 'info'; message: string }
