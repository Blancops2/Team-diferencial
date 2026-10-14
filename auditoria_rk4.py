"""Auditoria independiente. Ejecutar despues de experimentos_rk4.py.

Contrasta formulas, casos limite y solvers con soluciones exactas y DOP853.
Produce auditoria_resultados.json. No actualiza automaticamente las tablas LaTeX.
"""
import json
from fractions import Fraction as F
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from experimentos_rk4 import adaptive, fixed, rk4_step, euler_maruyama


def audit():
    checks = []
    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append(name)
    def raises(name, call, kind=ValueError):
        try:
            call()
        except kind:
            checks.append(name)
        else:
            raise AssertionError(name)

    A = np.array([[F(0),F(0),F(0),F(0)],
        [F(1,2),F(0),F(0),F(0)], [F(0),F(1,2),F(0),F(0)],
        [F(0),F(0),F(1),F(0)]], dtype=object)
    b = np.array([F(1,6),F(1,3),F(1,3),F(1,6)],dtype=object)
    c = A.sum(1)
    conditions = [b.sum(),b@c,b@(c*c),b@(A@c),b@(c*c*c),
        b@(c*(A@c)),b@(A@(c*c)),b@(A@A@c)]
    check("Ocho condiciones de orden con aritmetica racional",
          conditions == [F(1),F(1,2),F(1,3),F(1,6),F(1,4),F(1,8),F(1,12),F(1,24)])
    R = lambda z:1+z+z*z/2+z**3/6+z**4/24
    boundary = brentq(lambda z:R(z)-1,-3,-2)
    check("Frontera real de estabilidad RK4",abs(boundary+2.785293563405282)<1e-12)
    check("Polinomio de propagacion RK4",
        np.allclose(rk4_step(lambda t,y:-2*y,0,np.array([1.,3.]),0.2),R(-0.4)*np.array([1.,3.])))

    const = lambda t,y:np.array([2.,-3.])
    ts,ys,stats = adaptive(const,[1.,4.],0,1,h0=1,max_attempts=1)
    check("Finalizacion correcta en el ultimo intento permitido",
          ts[-1]==1 and np.allclose(ys[-1],[3.,1.]) and stats['nfev']==12)
    raises("Metodo fijo desconocido rechazado",lambda:fixed(lambda t,y:y,[1.],0,1,10,'typo'))
    raises("Numero de pasos nulo rechazado",lambda:fixed(lambda t,y:y,[1.],0,1,0))
    raises("Tolerancia NaN rechazada",lambda:adaptive(lambda t,y:y,[1.],0,1,rtol=np.nan))
    raises("Intervalo no finito rechazado",lambda:adaptive(lambda t,y:y,[1.],0,np.inf))
    raises("Forma incorrecta del campo rechazada",lambda:fixed(lambda t,y:1.,[1.,2.],0,1,10))
    raises("Difusion con dimensiones incorrectas rechazada",lambda:euler_maruyama(
        lambda t,x:x,lambda t,x:np.ones((2,1)),[1.,2.],0,1,np.zeros((4,2))))

    dw=np.array([[0.2,-0.1],[-0.3,0.4],[0.1,0.2]])
    drift=np.array([1.,-2.]);diff=np.array([[2.,1.],[-1.,3.]])
    value=euler_maruyama(lambda t,x:drift,lambda t,x:diff,[2.,3.],0,1,dw)
    exact=np.array([2.,3.])+drift+diff@dw.sum(0)
    check("Euler-Maruyama vectorial con difusion no nula",np.allclose(value,exact))

    # Estimador: exactitud asintotica de la correccion para y'=y.
    estimator_ratios=[]
    for h in [0.2,0.1,0.05]:
        y=np.array([1.]);coarse=rk4_step(lambda t,x:x,0,y,h)
        half=rk4_step(lambda t,x:x,0,y,h/2)
        fine=rk4_step(lambda t,x:x,h/2,half,h/2)
        estimator_ratios.append(float(((fine-coarse)/15/(np.exp(h)-fine))[0]))
    check("Estimador con factor 15 asintoticamente correcto",
          abs(estimator_ratios[-1]-1)<abs(estimator_ratios[0]-1)
          and abs(estimator_ratios[-1]-1)<0.08)

    # Ambas extensiones simultaneas: EDO no lineal, dato aleatorio y mallas propias.
    rng=np.random.default_rng(20261014);d=8;M=32
    q,_=np.linalg.qr(rng.normal(size=(d,d)))
    mat=q@np.diag(-np.linspace(.2,2,d))@q.T
    coupling=rng.normal(size=(d,d))/(4*np.sqrt(d));freq=np.linspace(.7,2,d)
    field=lambda t,y:mat@y+coupling@np.tanh(y)+.2*np.sin(freq*t)
    final_errors=[];counts=[];rejections=[];all_ratios=[]
    for z in rng.normal(size=(M,d)):
        t,y,st=adaptive(field,z,0,2,rtol=1e-7,atol=1e-10,h0=.1,hmax=.2)
        ref=solve_ivp(field,(0,2),z,method='DOP853',rtol=2e-12,atol=2e-14)
        check(f"Trayectoria aleatoria {len(counts)+1}: referencia y tiempo final",
              ref.success and t[-1]==2 and np.all(np.diff(t)>0))
        final_errors.append(float(np.max(np.abs(y[-1]-ref.y[:,-1]))))
        counts.append(st['accepted']);rejections.append(st['rejected']);all_ratios.append(st['max_estimator'])
        check(f"Trayectoria aleatoria {len(counts)}: contabilidad de evaluaciones",
              st['nfev']==12*(st['accepted']+st['rejected']))
    check("Ensayo conjunto: estimadores aceptados <=1",max(all_ratios)<=1)
    check("Ensayo conjunto: error final controlado en la familia ensayada",max(final_errors)<2e-6)
    result={"status":"passed","checks_passed":len(checks),"checks":checks,
        "rk4_conditions":[str(x) for x in conditions],"stability_boundary":boundary,
        "estimator_ratios":estimator_ratios,
        "joint_random_adaptive":{"seed":20261014,"M":M,"d":d,"rtol":1e-7,
            "atol":1e-10,"max_final_error":max(final_errors),
            "median_final_error":float(np.median(final_errors)),
            "accepted_min":min(counts),"accepted_max":max(counts),
            "total_rejections":sum(rejections),"max_scaled_estimator":max(all_ratios)}}
    Path(__file__).with_name('auditoria_resultados.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))


if __name__=='__main__':
    audit()
