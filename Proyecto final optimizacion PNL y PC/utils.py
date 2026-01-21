import numpy as np
from sympy import (
    N, lambdify, sin, cos, exp, tan, log, expand
)


def simbolos_ordenados(expr):# Obtiene símbolos libres de la expresión
    syms = list(expr.free_symbols)  
    return sorted(syms, key=lambda s: str(s))


def crear_funcion_numerica(expr, lista_vars):
    def f_num(punto):  #evalúa expr con valores numéricos 
        subs = {lista_vars[i]: punto[i] for i in range(len(lista_vars))} 
        return float(expr.subs(subs)) 
    return f_num 


def a_flotante_seguro(x, defecto=0.0):
    try:
        return float(x)
    except Exception:
        return defecto


def texto_ecuacion(expr):
    try:
        return str(expr)
    except Exception:
        return repr(expr)


def formatear_vector(v):
    try:
        return "[" + ", ".join(f"{float(x):.6g}" for x in v) + "]"
    except Exception:
        return str(v)


def formatear_matriz(M):
    try:
        return "[\n " + "\n ".join(
            "[" + ", ".join(f"{float(x):.6g}" for x in row) + "]" for row in M
        ) + "\n]"
    except Exception:
        return str(M)


def evaluar_numerico_seguro(expr, subs):
    """Evalúa expr con subs. Devuelve float o None si no es numérico real."""
    from sympy import N as symN
    try:  #sustituir y convertir a número
        val = expr.subs(subs)  # Aplica sustituciones
        val_num = symN(val)  # Aproxima
        if getattr(val_num, "free_symbols", None):  # Si quedan símbolos, no es numérico
            return None
        val_f = complex(val_num)  # Trata el complejo
        if abs(val_f.imag) > 1e-9:  # Rechaza imaginaria
            return None
        return float(val_f.real)
    except Exception:
        try:
            val_num = symN(expr.subs(subs))  # Reintento de conversión
            val_f = complex(val_num)  # Valor complejo
            if abs(val_f.imag) > 1e-9:  # Chequea parte imaginaria
                return None
            return float(val_f.real)
        except Exception:
            return None


def crear_funcion_numpy(expr, lista_vars):
    """Crea función numpy optimizada, con fallback a evaluación simbólica."""
    try:  # Intenta compilar función con numpy
        fn = lambdify(tuple(lista_vars), expr, modules=["numpy"])  # Genera función vectorizada

        def f_num(x):
            arr = np.asarray(x, dtype=float)
            return float(fn(*tuple(arr)))
        return f_num  # Devuelve versión numpy
    except Exception:
        def f_num(x):
            subs = {lista_vars[i]: x[i] for i in range(len(lista_vars))}  # Mapea variables a valores
            v = evaluar_numerico_seguro(expr, subs)
            if v is None:
                raise ValueError("No evaluable numéricamente")
            return v
        return f_num


def intentar_nsolve_vect(grads, lista_vars, log, nsolve_fn):
    """Fallback numérico: intenta nsolve con múltiples puntos iniciales."""
    soluciones = []
    semillas = [  # Lista de puntos iniciales
        tuple([0.0]*len(lista_vars)),
        tuple([1.0]*len(lista_vars)),
        tuple([-1.0]*len(lista_vars)),
        tuple([0.5]*len(lista_vars)),
        tuple([-0.5]*len(lista_vars))
    ]
    for seed in semillas:
        try:  #sistema no lineal
            sol = nsolve_fn(grads, tuple(lista_vars), seed,
                            tol=1e-14, maxsteps=200)
            if len(lista_vars) == 1:  # Normaliza salida a tupla
                sol_vec = (sol,)
            else:
                sol_vec = tuple(sol)
            sdict = {lista_vars[i]: sol_vec[i] for i in range(len(lista_vars))}  # Mapea a dict var->valor
            if sdict not in soluciones:  # Evita duplicados
                soluciones.append(sdict)
        except Exception:
            pass
    return soluciones 


def es_funcion_simple(expr, lista_vars):
    """
    CRITERIO PEDAGÓGICO para decidir Lagrange vs QP:
    - Simple: x**2 + y**2 (cuadrados puros, coef ≤10) → Lagrange más claro
    - Compleja: 3*x**2 + 2*x*y + 4*y**2 (términos cruzados o coef >10) → QP más eficiente
    """
    try:
        pol = expand(expr)  # Expande la expresión
        terminos = pol.as_ordered_terms()

        if any(expr.has(fn) for fn in (sin, cos, exp, tan, log)):  # Funciones no polinómicas, compleja
            return False

        try:  #grado máximo
            deg = max([
                t.as_poly(*lista_vars).total_degree() for t in terminos
            ]) if terminos else 0
            if deg > 2:  #no se considera simple
                return False
        except Exception:  # Si no puede calcular, considera compleja
            return False

        tiene_terminos_cruzados = False
        tiene_coeficientes_grandes = False

        for termino in terminos:
            try:
                poly = termino.as_poly(*lista_vars)  # Convierte a polinomio
                if poly is None:
                    continue  # Omite si no es

                coeff = abs(float(poly.coeffs()[0]))  # Toma coeficiente principal
                monomos = poly.monoms()

                if coeff > 10:
                    tiene_coeficientes_grandes = True

                for monom in monomos:  # Detecta términos cruzados de grado 2
                    if sum(monom) == 2 and 2 not in monom:
                        tiene_terminos_cruzados = True
            except Exception:
                continue

        return not (tiene_terminos_cruzados or tiene_coeficientes_grandes)

    except Exception:  # Ante fallo general, considera compleja
        return False
