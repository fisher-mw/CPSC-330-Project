import { useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Cell,
  ReferenceLine, Tooltip, ResponsiveContainer,
} from 'recharts'
import { predict } from '../api'

// ── Option lists ─────────────────────────────────────────────────
const PAY_OPTIONS = [
  { value: -2, label: 'No consumption' },
  { value: -1, label: 'Paid in full' },
  { value:  0, label: 'Revolving credit' },
  { value:  1, label: '1 month late' },
  { value:  2, label: '2 months late' },
  { value:  3, label: '3 months late' },
  { value:  4, label: '4 months late' },
  { value:  5, label: '5 months late' },
  { value:  6, label: '6 months late' },
  { value:  7, label: '7 months late' },
  { value:  8, label: '8 months late' },
  { value:  9, label: '9+ months late' },
]

const EDU_OPTIONS = [
  { value: 0, label: 'Unknown / Other' },
  { value: 1, label: 'High School' },
  { value: 2, label: 'University' },
  { value: 3, label: 'Graduate School' },
]

const SEX_OPTIONS    = [{ value: 1, label: 'Male' }, { value: 2, label: 'Female' }]
const MARRY_OPTIONS  = [{ value: 1, label: 'Married' }, { value: 2, label: 'Single' }, { value: 3, label: 'Other' }]

const MONTHS   = ['Sept', 'Aug', 'Jul', 'Jun', 'May', 'Apr']
const PAY_KEYS = ['pay_0', 'pay_2', 'pay_3', 'pay_4', 'pay_5', 'pay_6']
const BILL_KEYS   = ['bill_amt1', 'bill_amt2', 'bill_amt3', 'bill_amt4', 'bill_amt5', 'bill_amt6']
const PAYAMT_KEYS = ['pay_amt1',  'pay_amt2',  'pay_amt3',  'pay_amt4',  'pay_amt5',  'pay_amt6']

// ── Default form values (reasonable mid-range applicant) ─────────
const INIT = {
  limit_bal: 200000,
  sex: 2, education: 2, marriage: 1, age: 35,
  pay_0: 0, pay_2: 0, pay_3: 0, pay_4: 0, pay_5: -1, pay_6: -1,
  bill_amt1: 50000, bill_amt2: 48000, bill_amt3: 45000,
  bill_amt4: 42000, bill_amt5: 40000, bill_amt6: 38000,
  pay_amt1: 5000, pay_amt2: 5000, pay_amt3: 4000,
  pay_amt4: 4000, pay_amt5: 3000, pay_amt6: 3000,
}

// ── Custom SHAP tooltip ──────────────────────────────────────────
function ShapTooltip({ active, payload }) {
  if (!active || !payload?.length) return null
  const val = payload[0].value
  return (
    <div style={{
      background: '#1c1c1c', border: '1px solid #2d2d2d',
      borderRadius: 4, padding: '7px 11px', fontSize: 12, color: '#e8e8e8',
    }}>
      <span style={{ color: '#6a6a6a' }}>SHAP  </span>
      <span style={{ color: val > 0 ? '#d4d4d4' : '#8a8a8a' }}>
        {val > 0 ? '+' : ''}{val.toFixed(4)}
      </span>
    </div>
  )
}

// ── Main component ───────────────────────────────────────────────
export default function PredictorPage() {
  const [form, setForm]       = useState(INIT)
  const [result, setResult]   = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState(null)

  function handleChange(e) {
    const { name, value } = e.target
    setForm(prev => ({ ...prev, [name]: Number(value) }))
  }

  async function handleSubmit(e) {
    e.preventDefault()
    setLoading(true)
    setError(null)
    try {
      setResult(await predict(form))
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  // ── SHAP chart data: top 10, reversed so highest appears on top ──
  const shapData = result
    ? result.shap_breakdown
        .slice(0, 10)
        .map(item => ({ name: item.display_name, value: item.shap_value }))
        .reverse()
    : []
  const maxShap = shapData.length
    ? Math.max(...shapData.map(d => Math.abs(d.value))) * 1.25
    : 1

  return (
    <div className="page">
      <div className="predictor-layout">

        {/* ── Left: form ─────────────────────────────────────────── */}
        <form onSubmit={handleSubmit}>

          {/* Profile */}
          <section className="form-section">
            <h2 className="form-section-title">Profile</h2>
            <div className="field">
              <label>Credit Limit (TWD)</label>
              <input
                type="number" name="limit_bal"
                value={form.limit_bal} onChange={handleChange}
                min={10000} max={1000000} step={10000} required
              />
            </div>
            <div className="field-row">
              <div className="field">
                <label>Age</label>
                <input
                  type="number" name="age"
                  value={form.age} onChange={handleChange}
                  min={18} max={100} required
                />
              </div>
              <div className="field">
                <label>Sex</label>
                <select name="sex" value={form.sex} onChange={handleChange}>
                  {SEX_OPTIONS.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
                </select>
              </div>
            </div>
            <div className="field-row">
              <div className="field">
                <label>Education</label>
                <select name="education" value={form.education} onChange={handleChange}>
                  {EDU_OPTIONS.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
                </select>
              </div>
              <div className="field">
                <label>Marital Status</label>
                <select name="marriage" value={form.marriage} onChange={handleChange}>
                  {MARRY_OPTIONS.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
                </select>
              </div>
            </div>
          </section>

          {/* Repayment History */}
          <section className="form-section">
            <h2 className="form-section-title">Repayment History</h2>
            <p className="form-section-desc">Monthly payment status · Sept → Apr </p>
            <div className="field-grid-3">
              {PAY_KEYS.map((key, i) => (
                <div className="field" key={key}>
                  <label>{MONTHS[i]}</label>
                  <select name={key} value={form[key]} onChange={handleChange}>
                    {PAY_OPTIONS.map(o =>
                      <option key={o.value} value={o.value}>{o.label}</option>
                    )}
                  </select>
                </div>
              ))}
            </div>
          </section>

          {/* Bill Amounts */}
          <section className="form-section">
            <h2 className="form-section-title">Bill Amounts (TWD)</h2>
            <p className="form-section-desc">Statement balance each month · Sept → Apr </p>
            <div className="field-grid-3">
              {BILL_KEYS.map((key, i) => (
                <div className="field" key={key}>
                  <label>{MONTHS[i]}</label>
                  <input
                    type="number" name={key}
                    value={form[key]} onChange={handleChange}
                    step={1000}
                  />
                </div>
              ))}
            </div>
          </section>

          {/* Payment Amounts */}
          <section className="form-section">
            <h2 className="form-section-title">Payments Made (TWD)</h2>
            <p className="form-section-desc">Amount paid each month · Sept → Apr </p>
            <div className="field-grid-3">
              {PAYAMT_KEYS.map((key, i) => (
                <div className="field" key={key}>
                  <label>{MONTHS[i]}</label>
                  <input
                    type="number" name={key}
                    value={form[key]} onChange={handleChange}
                    min={0} step={500}
                  />
                </div>
              ))}
            </div>
          </section>

          {error && <div className="error-msg">{error}</div>}
          <button type="submit" className="btn-classify" disabled={loading}>
            {loading ? 'Classifying...' : 'Classify'}
          </button>
        </form>

        {/* ── Right: result panel ─────────────────────────────────── */}
        <div className="result-panel">
          {loading ? (
            <div className="result-empty">
              <div className="spinner" />
            </div>
          ) : !result ? (
            <div className="result-empty">
              Fill in the applicant profile<br />and click Classify.
            </div>
          ) : (
            <div className="result-card">

              {/* Verdict */}
              <div className={`result-verdict ${result.prediction === 1 ? 'verdict-default' : 'verdict-ok'}`}>
                {result.label}
              </div>

              {/* Probability bar */}
              <div className="result-prob">
                <div className="prob-row">
                  <span className="prob-label">Probability of default</span>
                  <span className="prob-value">
                    {(result.probability_default * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="prob-bar-track">
                  <div
                    className="prob-bar-fill"
                    style={{ width: `${result.probability_default * 100}%` }}
                  />
                </div>
              </div>

              {/* SHAP breakdown */}
              <div className="shap-header">
                <span className="shap-title">Top factors</span>
                <div className="shap-legend">
                  <span className="shap-legend-item">
                    <span className="shap-dot" style={{ background: '#d4d4d4' }} />
                    Risk ↑
                  </span>
                  <span className="shap-legend-item">
                    <span className="shap-dot" style={{ background: '#404040', border: '1px solid #555' }} />
                    Risk ↓
                  </span>
                </div>
              </div>

              <ResponsiveContainer width="100%" height={shapData.length * 30 + 30}>
                <BarChart
                  layout="vertical"
                  data={shapData}
                  margin={{ top: 0, right: 16, bottom: 0, left: 0 }}
                >
                  <XAxis
                    type="number"
                    domain={[-maxShap, maxShap]}
                    tickFormatter={v => v.toFixed(2)}
                    tick={{ fill: '#444', fontSize: 10 }}
                    axisLine={false}
                    tickLine={false}
                  />
                  <YAxis
                    type="category"
                    dataKey="name"
                    width={168}
                    tick={{ fill: '#7a7a7a', fontSize: 11 }}
                    axisLine={false}
                    tickLine={false}
                  />
                  <Tooltip content={<ShapTooltip />} cursor={{ fill: 'rgba(255,255,255,0.03)' }} />
                  <ReferenceLine x={0} stroke="#2d2d2d" strokeWidth={1} />
                  <Bar dataKey="value" radius={[0, 3, 3, 0]} maxBarSize={16}>
                    {shapData.map((entry, i) => (
                      <Cell
                        key={i}
                        fill={entry.value > 0 ? '#d4d4d4' : '#404040'}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>

            </div>
          )}
        </div>

      </div>
    </div>
  )
}
