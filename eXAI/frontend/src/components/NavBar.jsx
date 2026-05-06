import { NavLink } from 'react-router-dom'

export default function NavBar() {
  return (
    <nav className="navbar">
      <div className="navbar-inner">
        <span className="navbar-title">Credit Default Classifier</span>
        <div className="navbar-links">
          <NavLink
            to="/predict"
            className={({ isActive }) => 'nav-link' + (isActive ? ' active' : '')}
          >
            Classifier
          </NavLink>
          <NavLink
            to="/explain"
            className={({ isActive }) => 'nav-link' + (isActive ? ' active' : '')}
          >
            Explainability
          </NavLink>
        </div>
      </div>
    </nav>
  )
}
