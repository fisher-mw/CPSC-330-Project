const BASE = 'http://localhost:8000'

export async function predict(payload) {
  const res = await fetch(`${BASE}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || 'Prediction failed')
  }
  return res.json()
}

export async function getMetadata() {
  const res = await fetch(`${BASE}/metadata`)
  if (!res.ok) throw new Error('Failed to load metadata')
  return res.json()
}

export async function getEda() {
  const res = await fetch(`${BASE}/eda`)
  if (!res.ok) throw new Error('Failed to load EDA data')
  return res.json()
}
