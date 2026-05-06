import { useState, useEffect } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Cell,
  Tooltip, ResponsiveContainer,
} from 'recharts'
import { getMetadata, getEda } from '../api'

// ── Helpers ──────────────────────────────────────────────────────
function fmtK(v) {
  const n = Number(v)
  if (Math.abs(n) >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (Math.abs(n) >= 1_000)     return `${(n / 1_000).toFixed(0)}k`
  return n.toFixed(0)
}

function ChartTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div style={{
      background: '#1c1c1c', border: '1px solid #2d2d2d',
      borderRadius: 4, padding: '7px 12px', fontSize: 12, color: '#e8e8e8',
    }}>
      <div style={{ color: '#6a6a6a', marginBottom: 4 }}>{label}</div>
      {payload.map(p => (
        <div key={p.name} style={{ color: p.name === 'default' ? '#d4d4d4' : '#6a6a6a' }}>
          {p.name === 'default' ? 'Default' : 'No Default'}: {p.value.toLocaleString()}
        </div>
      ))}
    </div>
  )
}

function ImportanceTooltip({ active, payload }) {
  if (!active || !payload?.length) return null
  return (
    <div style={{
      background: '#1c1c1c', border: '1px solid #2d2d2d',
      borderRadius: 4, padding: '7px 12px', fontSize: 12, color: '#e8e8e8',
    }}>
      <span style={{ color: '#6a6a6a' }}>Mean |SHAP|  </span>
      {payload[0].value.toFixed(5)}
    </div>
  )
}

// ── Component ─────────────────────────────────────────────────────
export default function ExplainabilityPage() {
  const [meta, setMeta]   = useState(null)
  const [eda, setEda]     = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    Promise.all([getMetadata(), getEda()])
      .then(([m, e]) => { setMeta(m); setEda(e) })
      .catch(err => setError(err.message))
  }, [])

  if (error) return (
    <div className="page">
      <div className="api-error">
        <p>Cannot reach the API. Make sure the backend is running:</p>
        <code>uvicorn main:app --reload</code>
        <p className="error-detail">{error}</p>
      </div>
    </div>
  )

  if (!meta) return (
    <div className="page">
      <div className="loading-state">Loading...</div>
    </div>
  )

  const { metrics, class_distribution, shap_importance, model_info } = meta

  // Top 15 features for importance chart, reversed for top-to-bottom rendering
  const importanceData = shap_importance
    .slice(0, 15)
    .map(d => ({ name: d.display_name, importance: d.importance }))
    .reverse()

  return (
    <div className="page">

      {/* ── Header ─────────────────────────────────────────────── */}
      <section className="explain-section">
        <h1 className="page-title">Model Explainability</h1>
        <p className="page-desc">
          This classifier predicts whether a credit card client will default on
          their next payment. Because this decision directly affects a person's
          access to credit, understanding what drives each prediction is essential.
          The analysis below uses SHAP (SHapley Additive exPlanations) to expose
          the model's behavior at both the global and per-prediction level.
        </p>
      </section>

      {/* ── Model performance ──────────────────────────────────── */}
      <section className="explain-section">
        <h2 className="section-heading">Model Performance</h2>
        <p className="section-desc">
          {model_info.name} · evaluated on held-out test set vs. stratified dummy baseline
        </p>
        <div className="metrics-grid">
          {[
            { label: 'Accuracy',  val: metrics.accuracy,  base: metrics.baseline_accuracy },
            { label: 'F1 Score',  val: metrics.f1,        base: metrics.baseline_f1 },
            { label: 'Precision', val: metrics.precision,  base: null },
            { label: 'Recall',    val: metrics.recall,     base: null },
          ].map(m => (
            <div className="metric-card" key={m.label}>
              <span className="metric-label">{m.label}</span>
              <span className="metric-value">{m.val.toFixed(3)}</span>
              {m.base !== null && (
                <span className="metric-delta">
                  +{(m.val - m.base).toFixed(3)} vs baseline
                </span>
              )}
            </div>
          ))}
        </div>
      </section>

      {/* ── Class distribution ─────────────────────────────────── */}
      <section className="explain-section">
        <h2 className="section-heading">Class Distribution</h2>
        <p className="section-desc">Full dataset · 30,000 applicants</p>
        <div className="dist-bars">
          {[
            { label: 'No Default', pct: class_distribution.no_default, bg: '#383838' },
            { label: 'Default',    pct: class_distribution.default,    bg: '#d4d4d4' },
          ].map(b => (
            <div className="dist-bar-row" key={b.label}>
              <span className="dist-bar-label">{b.label}</span>
              <div className="dist-bar-track">
                <div
                  className="dist-bar-fill"
                  style={{ width: `${b.pct * 100}%`, background: b.bg }}
                />
              </div>
              <span className="dist-bar-pct">{(b.pct * 100).toFixed(1)}%</span>
            </div>
          ))}
        </div>
        <p className="note">
          The dataset is imbalanced (78 / 22). F1 score is the primary metric — it
          balances precision and recall and is more informative than accuracy alone.
        </p>
      </section>

      {/* ── Global feature importance ───────────────────────────── */}
      <section className="explain-section">
        <h2 className="section-heading">Global Feature Importance</h2>
        <p className="section-desc">
          Mean absolute SHAP value across 500 training samples · higher = more influential
        </p>
        <div className="chart-card">
          <ResponsiveContainer width="100%" height={importanceData.length * 26 + 40}>
            <BarChart
              layout="vertical"
              data={importanceData}
              margin={{ top: 5, right: 24, bottom: 5, left: 0 }}
            >
              <XAxis
                type="number"
                tick={{ fill: '#3d3d3d', fontSize: 10 }}
                axisLine={false}
                tickLine={false}
                tickFormatter={v => v.toFixed(3)}
              />
              <YAxis
                type="category"
                dataKey="name"
                width={200}
                tick={{ fill: '#7a7a7a', fontSize: 11 }}
                axisLine={false}
                tickLine={false}
              />
              <Tooltip content={<ImportanceTooltip />} cursor={{ fill: 'rgba(255,255,255,0.03)' }} />
              <Bar dataKey="importance" radius={[0, 3, 3, 0]} maxBarSize={14}>
                {importanceData.map((_, i) => (
                  <Cell
                    key={i}
                    fill={`hsl(0,0%,${28 + (i / importanceData.length) * 32}%)`}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </section>

      {/* ── Feature distributions ───────────────────────────────── */}
      {eda && (
        <section className="explain-section">
          <h2 className="section-heading">Feature Distributions</h2>
          <p className="section-desc">
            Training-set distributions split by default status
          </p>
          <div className="dist-chart-grid">
            {Object.entries(eda).map(([key, feature]) => (
              <div className="chart-card" key={key}>
                <p className="chart-title">{feature.label}</p>
                <ResponsiveContainer width="100%" height={190}>
                  <BarChart
                    data={feature.data}
                    margin={{ top: 4, right: 4, bottom: 24, left: -16 }}
                    barCategoryGap="20%"
                  >
                    <XAxis
                      dataKey="name"
                      tick={{ fill: '#484848', fontSize: 9 }}
                      axisLine={false}
                      tickLine={false}
                      interval={feature.type === 'histogram' ? 4 : 0}
                      tickFormatter={feature.type === 'histogram' ? fmtK : undefined}
                      angle={feature.type === 'histogram' ? -35 : 0}
                      textAnchor={feature.type === 'histogram' ? 'end' : 'middle'}
                    />
                    <YAxis
                      tick={{ fill: '#484848', fontSize: 9 }}
                      axisLine={false}
                      tickLine={false}
                      tickFormatter={v => v >= 1000 ? `${(v / 1000).toFixed(0)}k` : v}
                    />
                    <Tooltip
                      content={<ChartTooltip />}
                      cursor={{ fill: 'rgba(255,255,255,0.04)' }}
                    />
                    <Bar
                      dataKey="default"
                      name="default"
                      fill="#d4d4d4"
                      radius={[2, 2, 0, 0]}
                      maxBarSize={feature.type === 'bar' ? 28 : undefined}
                    />
                    <Bar
                      dataKey="no_default"
                      name="no_default"
                      fill="#303030"
                      radius={[2, 2, 0, 0]}
                      maxBarSize={feature.type === 'bar' ? 28 : undefined}
                    />
                  </BarChart>
                </ResponsiveContainer>
                <div className="chart-legend">
                  <span>
                    <span className="legend-dot" style={{ background: '#d4d4d4' }} />
                    Default
                  </span>
                  <span>
                    <span className="legend-dot" style={{ background: '#303030', border: '1px solid #484848' }} />
                    No Default
                  </span>
                </div>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* ── Why explainability matters ──────────────────────────── */}
      <section className="explain-section">
        <h2 className="section-heading">Why Explainability Matters</h2>
        <div className="text-section">
          <p>
            Credit default classification directly determines whether an individual
            can access loans, credit cards, or financial products. A black-box verdict
            — even an accurate one — fails the people it affects if they cannot
            understand or challenge its basis. Regulators in many jurisdictions now
            require that automated credit decisions be explainable.
          </p>
          <p>
            The SHAP analysis above shows that recent repayment behavior dominates the
            model's decisions. This is intuitively fair: a client who has consistently
            paid on time is objectively lower risk. Demographic features such as age,
            sex, and marital status contribute far less — a positive signal for model
            fairness compared to older, bias-prone scoring systems.
          </p>
          <p>
            Per-prediction SHAP breakdowns, visible on the Classifier page after each
            run, let an analyst explain exactly which factors drove a specific decision.
            This makes the model auditable and interpretable in contexts where decisions
            must be justified — whether to regulators, to the client, or to the institution.
          </p>
        </div>
      </section>

    </div>
  )
}
