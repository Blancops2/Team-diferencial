"""Solvers vectoriales y experimentos del informe RK4.

Python 3.10+, numpy, scipy. Ejecutar: python experimentos_rk4.py
Los resultados se guardan junto al script. Semilla fija y referencia independiente.
"""
from pathlib import Path
import json
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm


def _state(y0, *, vector=False):
    y = np.array(y0, dtype=float, copy=True)
    if y.size == 0 or y.ndim == 0 or (vector and y.ndim != 1):
        raise ValueError("El estado debe ser un array no vacio con forma valida")
    if not np.all(np.isfinite(y)):
        raise ValueError("Estado no finito")
    return y


def _field(f, t, y):
    value = np.asarray(f(t, y), dtype=float)
    if value.shape != y.shape:
        raise ValueError("El campo debe devolver la misma forma que el estado")
    if not np.all(np.isfinite(value)):
        raise FloatingPointError("Campo no finito")
    return value


def _interval(t0, tf):
    if not np.isfinite(t0) or not np.isfinite(tf) or tf <= t0:
        raise ValueError("Se requiere un intervalo finito con tf > t0")


def rk4_step(f, t, y, h):
    """RK4 clasico; f(t,y) devuelve un array de la misma forma que y."""
    y = np.asarray(y, dtype=float)
    k1 = h * _field(f, t, y)
    k2 = h * _field(f, t + h / 2, y + k1 / 2)
    k3 = h * _field(f, t + h / 2, y + k2 / 2)
    k4 = h * _field(f, t + h, y + k3)
    return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6


def fixed(f, y0, t0, tf, n, method="rk4"):
    _interval(t0, tf)
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n debe ser un entero positivo")
    if method not in ("rk4", "euler"):
        raise ValueError("Metodo desconocido: use rk4 o euler")
    y = _state(y0)
    h = (tf - t0) / n
    for j in range(n):
        t = t0 + j * h
        y = rk4_step(f, t, y, h) if method == "rk4" else y + h * _field(f, t, y)
        if not np.all(np.isfinite(y)):
            raise FloatingPointError("Estado no finito")
    return y


def adaptive(f, y0, t0, tf, *, rtol=1e-6, atol=1e-9,
             h0=0.05, hmax=np.inf, max_attempts=100000):
    """RK4 adaptativo por duplicacion. Acepta dos medios pasos, sin extrapolar.

    Norma infinito escalada, control I de exponente 1/5, 12 evaluaciones
    por intento. Integra hacia adelante y devuelve la malla aceptada.
    La tolerancia controla un estimador local, no certifica el error global.
    """
    _interval(t0, tf)
    y = _state(y0, vector=True)
    atol = np.broadcast_to(np.asarray(atol, float), y.shape)
    if (not np.isfinite(h0) or h0 <= 0 or np.isnan(hmax) or hmax <= 0
            or not np.isfinite(rtol) or rtol < 0
            or not np.all(np.isfinite(atol)) or np.any(atol <= 0)):
        raise ValueError("Intervalo, pasos o tolerancias invalidos")
    if (isinstance(max_attempts, (bool, np.bool_))
            or not isinstance(max_attempts, (int, np.integer)) or max_attempts < 1):
        raise ValueError("max_attempts debe ser un entero positivo")
    t, h = float(t0), min(h0, hmax, tf - t0)
    times, values, errors = [t], [y.copy()], []
    rejected = 0
    for attempt in range(max_attempts):
        h = min(h, hmax, tf - t)
        if t + h == t or h <= 0:
            raise RuntimeError("Paso menor que la resolucion de punto flotante")
        coarse = rk4_step(f, t, y, h)
        half = rk4_step(f, t, y, h / 2)
        fine = rk4_step(f, t + h / 2, half, h / 2)
        scale = atol + rtol * np.maximum(np.abs(y), np.abs(fine))
        ratio = np.max(np.abs((fine - coarse) / 15) / scale)
        if not np.isfinite(ratio) or not np.all(np.isfinite(fine)):
            raise FloatingPointError("Campo o estimador no finito")
        accepted = ratio <= 1
        if accepted:
            t = tf if h == tf - t else t + h
            y = fine
            times.append(t)
            values.append(y.copy())
            errors.append(float(ratio))
            if t >= tf:
                return np.array(times), np.array(values), {
                    "accepted": len(times) - 1, "rejected": rejected,
                    "nfev": 12 * (attempt + 1), "max_estimator": max(errors)}
        else:
            rejected += 1
        factor = 5.0 if ratio == 0 else np.clip(0.9 * ratio ** (-0.2), 0.2, 5.0)
        if not accepted:
            factor = min(factor, 1.0)
        h *= factor
    raise RuntimeError("Numero maximo de intentos alcanzado")


def euler_maruyama(a, b, x0, t0, tf, increments):
    """EDS vectorial de Ito con incrementos dados (n,m).

    a(t,x): (d,), b(t,x): (d,m). Malla uniforme determinista.
    El llamador genera los incrementos y controla el acoplamiento browniano.
    """
    _interval(t0, tf)
    dw = np.asarray(increments, float)
    if dw.ndim != 2 or min(dw.shape) < 1 or not np.all(np.isfinite(dw)):
        raise ValueError("Incrementos finitos con forma (n,m), n,m positivos")
    x = _state(x0, vector=True)
    h = (tf - t0) / len(dw)
    for n, delta in enumerate(dw):
        t = t0 + n * h
        drift = _field(a, t, x)
        diffusion = np.asarray(b(t, x), dtype=float)
        if diffusion.shape != (len(x), dw.shape[1]):
            raise ValueError("La difusion debe tener forma (d,m)")
        if not np.all(np.isfinite(diffusion)):
            raise FloatingPointError("Difusion no finita")
        x += h * drift + diffusion @ delta
        if not np.all(np.isfinite(x)):
            raise FloatingPointError("Estado no finito")
    return x


def run():
    rng = np.random.default_rng(20261013)
    d, T = 8, 2.0
    q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    A = q @ np.diag(-np.linspace(0.2, 2.0, d)) @ q.T
    B = rng.normal(size=(d, d)) / (4 * np.sqrt(d))
    w = np.linspace(0.7, 2.0, d)
    f = lambda t, y: A @ y + B @ np.tanh(y) + 0.2 * np.sin(w * t)
    y0 = rng.normal(size=d)
    ref = solve_ivp(f, (0, T), y0, method="DOP853", rtol=2e-13, atol=2e-15)
    ref2 = solve_ivp(f, (0, T), y0, method="DOP853", rtol=5e-14, atol=5e-16)
    assert ref.success and ref2.success
    reference_difference = float(np.linalg.norm(ref.y[:, -1] - ref2.y[:, -1], np.inf))
    order = []
    for n in [20, 40, 80, 160]:
        e = float(np.linalg.norm(fixed(f, y0, 0, T, n, "euler") - ref2.y[:, -1], np.inf))
        r = float(np.linalg.norm(fixed(f, y0, 0, T, n) - ref2.y[:, -1], np.inf))
        order.append({"n": n, "h": T/n, "euler": e, "rk4": r})
    for i, row in enumerate(order):
        row["p_euler"] = None if i == 0 else float(np.log2(order[i-1]["euler"] / row["euler"]))
        row["p_rk4"] = None if i == 0 else float(np.log2(order[i-1]["rk4"] / row["rk4"]))
    assert order[-1]["p_rk4"] > 3.8 and 0.9 < order[-1]["p_euler"] < 1.2

    # Familia lineal no autonoma: solucion exacta para cualquier dato inicial.
    # Un pulso suave obliga al controlador a refinar cerca de t=1.
    amps = np.linspace(0.2, 0.6, d)
    v = np.ones(d) / np.sqrt(d)
    def g(t):
        return amps * np.sin(w*t) + v * np.exp(-((t-1)/0.12)**2)
    def gp(t):
        pulse = np.exp(-((t-1)/0.12)**2)
        return amps*w*np.cos(w*t) - v * (2*(t-1)/0.12**2)*pulse
    fm = lambda t, y: A @ (y-g(t)) + gp(t)
    exact = lambda t, z: g(t) + expm(A*t) @ (z-g(0))
    adaptive_rows = []
    mesh = None
    for tol in [1e-3, 1e-5, 1e-7, 1e-9]:
        ts, ys, stats = adaptive(fm, y0, 0, T, rtol=tol, atol=tol*1e-3,
                                 h0=0.3, hmax=0.1)
        err = max(np.linalg.norm(y-exact(t, y0), np.inf) for t, y in zip(ts, ys))
        adaptive_rows.append({"tol":tol, "error":float(err), **stats})
        if tol == 1e-7:
            mesh = [[float(t),float(h)] for t,h in zip(ts[:-1],np.diff(ts))]
    assert adaptive_rows[-1]["error"] < adaptive_rows[0]["error"]
    assert all(x["max_estimator"] <= 1 for x in adaptive_rows)

    # Monte Carlo emparejado: separar error numerico del error de muestreo.
    M = 20000
    mu = np.linspace(-0.4, 0.5, d)
    C = rng.normal(size=(d, d)) / np.sqrt(d)
    Sigma = 0.04 * C @ C.T
    Z = rng.multivariate_normal(mu, Sigma, size=M)
    P = expm(A*T)
    exact_ensemble = Z @ P.T
    # Vectorizacion sobre M trayectorias con el mismo RK4 de dimension d.
    approximation = fixed(lambda t,Y:Y@A.T, Z, 0, T, 40)
    true_mean, true_cov = P @ mu, P @ Sigma @ P.T
    ensemble = {
        "M": M,
        "paired_rms": float(np.sqrt(np.mean(np.sum((approximation-exact_ensemble)**2,axis=1)))),
        "numerical_mean": float(np.linalg.norm(approximation.mean(0)-exact_ensemble.mean(0))),
        "sampling_mean": float(np.linalg.norm(exact_ensemble.mean(0)-true_mean)),
        "total_mean": float(np.linalg.norm(approximation.mean(0)-true_mean)),
        "sampling_covariance": float(np.linalg.norm(np.cov(exact_ensemble,rowvar=False)-true_cov,"fro")),
        "numerical_covariance": float(np.linalg.norm(np.cov(approximation,rowvar=False)-np.cov(exact_ensemble,rowvar=False),"fro"))}
    assert ensemble["numerical_mean"] < ensemble["sampling_mean"]

    # Verificacion estocastica: movimiento browniano geometrico solo como test.
    # Los solvers anteriores siguen siendo generales. Mismas trayectorias W
    # para cada refinamiento, sin volver a generar ruido al cambiar de malla.
    a, b, M_sde, finest = -0.3, 0.7, 12000, 1024
    dw = rng.normal(scale=np.sqrt(1/finest), size=(M_sde,finest))
    exact_sde = np.exp((a-b*b/2) + b*dw.sum(axis=1))
    stochastic = []
    for n in [32,64,128,256,512,1024]:
        coarse_dw = dw.reshape(M_sde,n,finest//n).sum(axis=2)
        X = np.prod(1+a/n+b*coarse_dw,axis=1)
        stochastic.append({"n":n,"h":1/n,
            "rmse":float(np.sqrt(np.mean((X-exact_sde)**2))),
            "weak_exact_bias":float(abs((1+a/n)**n-np.exp(a)))})
    for j,row in enumerate(stochastic):
        row["order"] = None if j == 0 else float(np.log2(stochastic[j-1]["rmse"]/row["rmse"]))
    # Un test independiente del solver EM vectorial sin difusion.
    zero = np.zeros((100,1))
    em = euler_maruyama(lambda t,x:A@x,lambda t,x:np.zeros((d,1)),y0,0,T,zero)
    assert np.allclose(em,fixed(lambda t,x:A@x,y0,0,T,100,"euler"))
    result = {"seed":20261013,"dimension":d,"T":T,
        "reference_difference":reference_difference,"convergence":order,
        "adaptive":adaptive_rows,"mesh":mesh,"ensemble":ensemble,
        "stochastic":stochastic,"versions":{"numpy":np.__version__}}
    import scipy
    result["versions"]["scipy"]=scipy.__version__
    target=Path(__file__).with_name("resultados.json")
    target.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k != "mesh"},indent=2))


if __name__ == "__main__":
    run()
