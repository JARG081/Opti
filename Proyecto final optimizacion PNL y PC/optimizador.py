import numpy as np
import itertools
import traceback

from sympy import (
    symbols, Eq, solve, diff, Matrix, sympify, expand, N, Symbol,
    sin, cos, exp, tan, log, nsolve, linear_eq_to_matrix
)

try:
    import scipy.optimize as sco
except Exception:
    sco = None

from utils import (
    simbolos_ordenados, crear_funcion_numerica, a_flotante_seguro,
    texto_ecuacion, formatear_vector, formatear_matriz,
    evaluar_numerico_seguro, crear_funcion_numpy,
    intentar_nsolve_vect, es_funcion_simple
)


class Optimizador:
    def __init__(self):
        pass

    def sugerir_metodo(self, expr_func, lineas_restr, lista_vars):
        nonpoly_funcs = any(expr_func.has(fn) for fn in (sin, cos, exp, tan, log))  #funciones no polinómicas

        try:  # Analiza grado polinómico de la función objetivo
            pol = expand(expr_func)
            terminos = pol.as_ordered_terms()
            deg = max([t.as_poly(*lista_vars).total_degree()
                       for t in terminos]) if terminos else 0
            es_cuad = deg <= 2  # Es cuadrática
        except Exception:
            es_cuad = False

        func_simple = es_funcion_simple(expr_func, lista_vars)  # Evalúa simplicidad
        any_ineq = any(('>=' in r or '<=' in r) for r in lineas_restr)
        any_eq = any('=' in r for r in lineas_restr)

        if not lineas_restr:
            if nonpoly_funcs:
                return "Hessiana"
            return "Sin restricciones" if es_cuad else "Hessiana"  # Cuadrática → análisis sin restricciones

        if any_eq and not any_ineq:  # Solo igualdades
            if func_simple:
                return "Lagrange (igualdad)"
            if es_cuad:
                return "Programación Cuadrática"
            return "Lagrange (igualdad)"

        if any_ineq:
            return "KKT (desigualdad)"

        return "Programación Cuadrática" if es_cuad else "KKT (desigualdad)"


    def resolver(self, texto_func, lineas_restr, parametros_grad, metodo_manual,
                 usar_sugerido, log, mostrar_error):
        """
        texto_func: str con la función objetivo
        lineas_restr: lista[str] con restricciones
        parametros_grad: dict con alpha, max_iter, tol, init
        metodo_manual: str del OptionMenu
        usar_sugerido: bool
        log: función que recibe un str y lo imprime en el panel
        mostrar_error: función para mostrar errores (ej: messagebox.showerror)
        """
        paso = 1  # Contador de pasos para trazas

        def log_int(msg=""):
            nonlocal paso 
            log(f"[Paso {paso}] {msg}")
            paso += 1

        if not texto_func:  # Valida que haya función
            mostrar_error("Error", "Función objetivo vacía.")
            return

        try:  # Convierte texto a expresión simbólica
            expr = sympify(texto_func)
        except Exception as e:
            mostrar_error("Error", f"Expresión objetivo inválida: {e}")
            return

        lista_vars = simbolos_ordenados(expr)  # Detecta variables en la función
        if not lista_vars:
            lista_vars = [Symbol('x'), Symbol('y')]

        log_int(f"Variables detectadas: {', '.join(str(v) for v in lista_vars)}")
        log_int(f"Función: {texto_ecuacion(expr)}")

        igualdades = []
        desigualdades = []
        for r in lineas_restr:
            limpio = r.replace(" ", "")
            if "=" in limpio and ">=" not in limpio and "<=" not in limpio:  # Igualdad pura
                lhs, rhs = limpio.split("=", 1)
                try:
                    igualdades.append(Eq(sympify(lhs), sympify(rhs)))  # Crea Eq simbólica
                except Exception as e:
                    mostrar_error("Error", f"Restricción '{r}' inválida: {e}")
                    return
            elif ">=" in limpio or "<=" in limpio:
                desigualdades.append(r)
            else:
                mostrar_error(
                    "Error",
                    f"Restricción '{r}' debe contener '=' o '>=' o '<='."
                )
                return

        log_int(f"Igualdades: {len(igualdades)}, Desigualdades: {len(desigualdades)}")

        sugerido = self.sugerir_metodo(expr, lineas_restr, lista_vars)  # Obtiene sugerencia automática
        log(f"Sugerido: {sugerido}\n")

        if usar_sugerido or metodo_manual == "Auto":  # Decide método a usar
            metodo = sugerido
        else:
            metodo = metodo_manual

        log_int(f"Método seleccionado: {metodo}")

        try:  # Ejecuta método seleccionado
            if metodo == "Sin restricciones":
                self._decidir_sin_restricciones(expr, lista_vars, log_int)
            elif metodo == "Hessiana":
                self._resolver_hessiana(expr, lista_vars, log_int)
            elif metodo.startswith("Lagrange"):
                self._resolver_lagrange(expr, lista_vars, igualdades, log_int)
            elif metodo.startswith("KKT"):
                self._resolver_kkt_conjuntos_activos(
                    expr, lista_vars, igualdades, desigualdades, log_int
                )
            elif metodo.startswith("Gradiente"):
                self._resolver_gradiente(expr, lista_vars, parametros_grad, log_int)
            elif metodo.startswith("Programación Cuadrática"):
                self._resolver_qp_general(
                    expr, lista_vars, igualdades, desigualdades, parametros_grad, log_int
                )
            else:
                log_int("Método no implementado.")
        except Exception as e:
            tb = traceback.format_exc()
            mostrar_error("Error en resolución", f"{e}\n\n{tb}")

    def _decidir_sin_restricciones(self, f, lista_vars, log):
        log("Caso sin restricciones: intento resolver simbólicamente y analizar Hessiana.")  # Explica
        grads = [diff(f, v) for v in lista_vars]  # Calcula gradientes
        for i, g in enumerate(grads):  # Muestra gradiente
            log(f"∂f/∂{lista_vars[i]} = {texto_ecuacion(g)}")
        try:  # Intenta resolver simbólicamente
            soluciones = solve(grads, tuple(lista_vars), dict=True)
        except Exception:
            soluciones = []
        if not soluciones:
            soluciones = intentar_nsolve_vect(grads, lista_vars, log, nsolve)

        if not soluciones:  # Si aún no hay soluciones, usa gradiente
            log("No se hallaron soluciones simbólicas. Uso método numérico (gradiente).")
            parametros_grad = {
                "alpha": 1.0,
                "max_iter": 1000,
                "tol": 1e-6,
                "init": ",".join(["0.0"] * len(lista_vars))
            }
            return self._resolver_gradiente(f, lista_vars, parametros_grad, log)

        H = Matrix([[diff(f, vi, vj) for vi in lista_vars] for vj in lista_vars])  # Construye Hessiana simbólica
        log("Hessiana simbólica (una línea): " + texto_ecuacion(H))

        debe_hessiana = False  # Indica si se requiere análisis adicional
        for s in soluciones:  # Evalúa cada punto crítico
            log(f"Punto crítico candidato: {s}")
            try:
                H_e = H.subs(s)  # Sustituye punto en Hessiana
                try:
                    eigs = list(H_e.eigenvals().keys())
                    eigs_num = []  #números reales
                    for ev in eigs:
                        try:
                            evn = complex(N(ev))  # Aproxima numéricamente
                            eigs_num.append(evn.real)  # Usa parte real
                        except Exception:
                            eigs_num.append(None)
                    log(f"Valores propios (num aproximados): {eigs_num}")
                    if all(ev is not None and ev > 1e-9 for ev in eigs_num):
                        log("Clasificación en este punto: mínimo local (H positiva definida).")
                    elif all(ev is not None and ev < -1e-9 for ev in eigs_num):
                        log("Clasificación en este punto: máximo local (H negativa definida).")
                    else:  # Indefinida o mixta
                        log("Clasificación en este punto: indefinido / punto de silla (Hessiana mixta).")
                        debe_hessiana = True
                except Exception:
                    log("No fue posible obtener valores propios numéricos de la Hessiana en este punto.")
                    debe_hessiana = True
            except Exception as e:
                log(f"Error evaluando Hessiana en {s}: {e}")
                debe_hessiana = True

        if debe_hessiana:
            log("Al menos un punto crítico resultó en Hessiana indefinida o no evaluable: ejecuto análisis por Hessiana.")
            return self._resolver_hessiana(f, lista_vars, log)

        log("\n=== RESULTADO FINAL ===")
        for s in soluciones:  # Para cada punto, evalúa y clasifica
            val_f = evaluar_numerico_seguro(f, s)
            log(f"\nPunto: {s}")
            if val_f is None:  # Si no es numérico, muestra simbólico
                log(f"f (no numérico): {texto_ecuacion(f.subs(s))}")
            else:
                log(f"f = {val_f:.12g}")

            try:
                H_e = H.subs(s)  #Hessiana
                eigs = list(H_e.eigenvals().keys())  # Valores propios simbólicos
                eigs_num = []
                for ev in eigs:
                    try:
                        evn = complex(N(ev))  # Aproximación
                        eigs_num.append(evn.real)
                    except Exception:
                        eigs_num.append(None)

                if all(ev is not None and ev > 1e-9 for ev in eigs_num):  # Positiva definida
                    log("Tipo: MÍNIMO LOCAL (H positiva definida)")
                elif all(ev is not None and ev < -1e-9 for ev in eigs_num):  # Negativa definida
                    log("Tipo: MÁXIMO LOCAL (H negativa definida)")
                else:  # Indefinida
                    log("Tipo: PUNTO DE SILLA (H indefinida)")
            except Exception:
                pass

    def _resolver_hessiana(self, f, lista_vars, log):
        log("Cálculo de Hessiana y clasificación de puntos críticos.")
        grads = [diff(f, v) for v in lista_vars]  #parciales
        H = Matrix([[diff(f, vi, vj) for vi in lista_vars] for vj in lista_vars])  # Construcción de Hessiana
        for i, g in enumerate(grads):  # Muestra gradiente
            log(f"∂f/∂{lista_vars[i]} = {texto_ecuacion(g)}")
        log("Matriz Hessiana simbólica (sin formato multilinea):")
        log(texto_ecuacion(H))
        try:  # Resolución simbólica
            soluciones = solve(grads, tuple(lista_vars), dict=True)
        except Exception:
            soluciones = []
        if not soluciones:
            soluciones = intentar_nsolve_vect(grads, lista_vars, log, nsolve)
        if not soluciones:
            log("No se encontraron puntos críticos simbólicos.")
            return

        log("\n=== RESULTADO FINAL ===")
        for s in soluciones:
            H_evaluada = H.subs(s)  # Sustituye punto en Hessiana
            log(f"\nPunto crítico: {s}")
            val_f = evaluar_numerico_seguro(f, s)
            if val_f is None:
                log(f"f (no numérico): {texto_ecuacion(f.subs(s))}")
            else:
                log(f"f = {val_f:.12g}")

            try:
                eigs = list(H_evaluada.eigenvals().keys())  # valores simbólicos
                eigs_num = [complex(N(ev)) for ev in eigs]
                log(f"Valores propios H: {[float(ev.real) for ev in eigs_num]}")
                if all(ev.real > 1e-9 for ev in eigs_num):  # Positiva definida
                    log("Tipo: MÍNIMO LOCAL (H positiva definida)")
                elif all(ev.real < -1e-9 for ev in eigs_num):  # Negativa definida
                    log("Tipo: MÁXIMO LOCAL (H negativa definida)")
                else:  # Indefinida
                    log("Tipo: PUNTO DE SILLA o INDEFINIDO")
            except Exception:
                log("No fue posible calcular valores propios numéricos.")

    def _resolver_lagrange(self, f, lista_vars, igualdades, log):
        if not igualdades:
            raise ValueError("No hay igualdades para Lagrange.")
        log("Construyendo Lagrangiano con igualdades.")
        n_eq = len(igualdades)
        lam = symbols(f'lam0:{n_eq}')  # Multiplicadores de Lagrange
        exprs_constr = [(eq.lhs - eq.rhs) for eq in igualdades]
        L = f - sum(lam[i] * exprs_constr[i] for i in range(n_eq))
        log("Lagrangiano (una sola línea, sin exponentes multilinea):")
        log("L = " + texto_ecuacion(L))
        sistema = [diff(L, v) for v in lista_vars] + [diff(L, lm) for lm in lam]
        log("Ecuaciones del sistema (gradientes y derivadas respecto multiplicadores):")
        for i, e in enumerate(sistema):
            log(f"{i+1}: {texto_ecuacion(e)}")
        incognitas = tuple(lista_vars) + tuple(lam)
        soluciones = solve(sistema, incognitas, dict=True)
        if not soluciones:  # Maneja ausencia de solución simbólica
            log("No se hallaron soluciones simbólicas para el sistema de Lagrange.")
            return

        H_f = Matrix([[diff(f, vi, vj) for vi in lista_vars] for vj in lista_vars])  # Hessiana de f
        A = Matrix([[diff(expr, v) for v in lista_vars] for expr in exprs_constr])  # Jacobiana de restricciones

        log("\n=== RESULTADO FINAL ===")
        for s in soluciones:
            log(f"\nPunto: {s}")
            fval = evaluar_numerico_seguro(f, s)
            if fval is None:
                log(f"f (no numérico): {texto_ecuacion(f.subs(s))}")
            else:
                log(f"f = {fval:.12g}")

            try:  # Construye y evalúa Hessiana bordeada para clasificación
                H_eval = H_f.subs(s)
                A_eval = A.subs(s)
                n = len(lista_vars)
                m = len(igualdades)
                H_bordeada = Matrix.zeros(n + m, n + m)  # Matriz bordeada
                H_bordeada[m:, m:] = H_eval  # Bloque Hessiano
                H_bordeada[:m, m:] = A_eval  # Bloque Jacobiano
                H_bordeada[m:, :m] = A_eval.T  # Transpuesta en bloque simétrico

                eigs_bordeada = list(H_bordeada.eigenvals().keys())
                eigs_num = []
                for ev in eigs_bordeada: 
                    try:
                        evn = complex(N(ev))
                        eigs_num.append(evn.real)
                    except Exception:
                        pass

                if eigs_num:
                    log(f"Valores propios Hessiana bordeada: {[round(e, 6) for e in sorted(eigs_num)]}")
                    eigs_sorted = sorted(eigs_num)
                    relevantes = eigs_sorted[-(n-m):] if n > m else eigs_sorted  # Selección relevante

                    if all(e > 1e-9 for e in relevantes):  # Mínimo condicionado
                        log("Tipo: MÍNIMO LOCAL condicionado")
                    elif all(e < -1e-9 for e in relevantes):  # Máximo condicionado
                        log("Tipo: MÁXIMO LOCAL condicionado")
                    else:  # Indefinido/silla
                        log("Tipo: PUNTO DE SILLA o INDEFINIDO")
            except Exception as e:
                log(f"No se pudo evaluar Hessiana bordeada: {e}")

    def _parsear_inecuacion(self, ineq):
        r = ineq.replace(" ", "")
        if ">=" in r:
            lhs, rhs = r.split(">=", 1)
            return sympify(lhs) - sympify(rhs)
        elif "<=" in r:
            lhs, rhs = r.split("<=", 1)
            return sympify(rhs) - sympify(lhs)
        else:
            raise ValueError("Inecuación malformada.")

    def _resolver_kkt_conjuntos_activos(self, f, lista_vars, igualdades, desigualdades, log):
        log("Resolviendo condiciones KKT por enumeración de conjuntos activos (active-set).")
        g_exprs = []  #desigualdades
        for r in desigualdades:  # Parsea
            g_exprs.append(self._parsear_inecuacion(r))
        log(f"Igualdades: {len(igualdades)}, Desigualdades procesadas: {len(g_exprs)}")

        max_activos = 3
        if len(g_exprs) > 10:
            raise ValueError("Demasiadas desigualdades para enumerar (limite práctico).")

        encontrado = False  # Bandera de solución factible
        indices = list(range(len(g_exprs)))
        for r in range(0, min(len(g_exprs), max_activos) + 1):
            for subconjunto in itertools.combinations(indices, r):
                activos = [g_exprs[i] for i in subconjunto]
                lam_eq = symbols(f'lamE0:{len(igualdades)}')  # Multiplicadores de igualdades
                mu_act = symbols(f'muA0:{len(activos)}')  # Multiplicadores de activos
                Ls = f
                for i_eq, eq in enumerate(igualdades):  # Añade igualdades
                    Ls = Ls - lam_eq[i_eq] * (eq.lhs - eq.rhs)
                for j, g in enumerate(activos):
                    Ls = Ls - mu_act[j] * g
                sistema = [diff(Ls, v) for v in lista_vars]
                for eq in igualdades:
                    sistema.append(eq.lhs - eq.rhs)
                for g in activos:
                    sistema.append(g)
                desconocidas = tuple(lista_vars) + tuple(lam_eq) + tuple(mu_act)#variables, producto lambda y producto miu
                try:
                    soluciones = solve(sistema, desconocidas, dict=True)
                except Exception:
                    soluciones = []
                if not soluciones:
                    continue
                for s in soluciones:
                    log(f"Solución cruda: {s}")
                    residuos = []  # Valores g(x)
                    factible = True
                    for g in g_exprs:
                        r_val = evaluar_numerico_seguro(g, s)
                        residuos.append(r_val)
                        if r_val is None or r_val < -1e-8:
                            factible = False

                    mu_vals = []
                    mu_no_neg = True
                    for j in range(len(activos)):
                        raw = s.get(mu_act[j], 0)  # Obtiene valor simbólico
                        mu_num = None
                        try:
                            if raw is None:
                                mu_num = 0.0  # Trata None como 0
                            else:
                                mu_eval = N(raw)  # Aproxima numéricamente
                                mu_c = complex(mu_eval)  # Convierte a complejo
                                if abs(mu_c.imag) > 1e-8:
                                    mu_num = None
                                else:
                                    mu_num = float(mu_c.real)
                        except Exception:
                            mu_num = None
                        mu_vals.append(mu_num)
                        if mu_num is None or mu_num < -1e-9:
                            mu_no_neg = False

                    log(f"Residuos desigualdades (g): {residuos}")
                    log(f"Mu (multiplicadores asociados a activos): {mu_vals}")
                    log(f"Activos={subconjunto}, factible={factible}, mu_no_neg={mu_no_neg}")
                    if factible and mu_no_neg:  # Condiciones KKT satisfechas
                        encontrado = True
                        val = evaluar_numerico_seguro(f, s) 
                        log(f"\n=== SOLUCIÓN KKT FACTIBLE (activos={subconjunto}) ===")
                        log(f"Punto: {s}")
                        if val is None:
                            log(f"f (no numérico): {texto_ecuacion(f.subs(s))}")
                        else:
                            log(f"f = {val:.12g}")
                        log(f"Multiplicadores mu: {mu_vals}")
        if not encontrado:
            log("\nNo se hallaron soluciones KKT factibles por enumeración. Se recomienda usar solver numérico (SLSQP) como alternativa.")

    def _gradientes_simbolicos(self, f, lista_vars):
        return [diff(f, v) for v in lista_vars]

    def _evaluar_grad(self, grad_sim, punto, lista_vars):
        subs = {lista_vars[i]: punto[i] for i in range(len(lista_vars))} #soluciones
        vals = []  #componentes de gradiente
        for g in grad_sim:
            v = evaluar_numerico_seguro(g, subs)
            if v is None:
                try:
                    v = float(N(g.subs(subs)))
                except Exception:
                    v = 0.0
            vals.append(v)
        return np.array(vals, dtype=float)  # Devuelve gradiente como array

    def _resolver_gradiente(self, f, lista_vars, parametros_grad, log):
        alpha0 = a_flotante_seguro(parametros_grad.get("alpha", 1.0), 1.0)
        max_iter = int(a_flotante_seguro(parametros_grad.get("max_iter", 1000), 1000))
        tol = a_flotante_seguro(parametros_grad.get("tol", 1e-6), 1e-6)  # Tolerancia
        init_text = parametros_grad.get("init", "").strip()
        try:
            init_vals = [float(v.strip()) for v in init_text.split(",") if v.strip() != '']
        except Exception:
            init_vals = []
        if len(init_vals) < len(lista_vars):
            init_vals = init_vals + [0.0] * (len(lista_vars) - len(init_vals))
        xk = np.array(init_vals[:len(lista_vars)], dtype=float)
        log(f"Gradiente numérico (Armijo). Inicio={formatear_vector(xk)}, alpha0={alpha0}, max_iter={max_iter}, tol={tol}")
        grad_sim = self._gradientes_simbolicos(f, lista_vars)
        f_num = crear_funcion_numerica(f, lista_vars)

        historia = []
        c = 1e-4
        rho = 0.5  # Factor de reducción del paso
        for k in range(1, max_iter+1):
            gval = self._evaluar_grad(grad_sim, xk, lista_vars)  # Evalúa gradiente en xk
            ng = np.linalg.norm(gval)
            fv = f_num(xk)  # Valor de la función
            historia.append((k, xk.copy(), fv, ng))
            if ng < tol:
                log(f"Convergencia alcanzada: ||grad|| = {ng:.6g} < tol")
                break
            pk = -gval
            alpha = alpha0  # Inicializa paso
            while alpha > 1e-12:  # Búsqueda de línea con Armijo
                xprueba = xk + alpha * pk
                fprueba = f_num(xprueba)
                if fprueba <= fv + c * alpha * np.dot(gval, pk):
                    break
                alpha *= rho  # Reduce paso
            xk = xk + alpha * pk 
            if k % 50 == 0:
                log(f"Iter {k}: x={formatear_vector(xk)}, f={fv:.6g}, ||g||={ng:.3g}, alpha={alpha:.3g}")
        for it, pt, fv, ng in historia[-20:]:
            log(f"Iter {it}: x={formatear_vector(pt)}, f={fv:.9g}, ||g||={ng:.3g}")
        log(f"Resultado final: x={formatear_vector(xk)}, f={f_num(xk):.9g}")

    def _resolver_qp_general(self, f, lista_vars, igualdades, desigualdades,
                             parametros_grad, log):
        log("Intentando interpretar función como cuadrática (QP).")
        pol = expand(f)
        n = len(lista_vars)  # Número de variables
        Q = np.zeros((n, n), dtype=float)  # Matriz cuadrática
        c_vec = np.zeros((n,), dtype=float)  # Vector lineal
        const = 0.0
        for termino in pol.as_ordered_terms():
            poly = termino.as_poly(*lista_vars)  # Convierte a polinomio
            if poly is None:
                continue
            monomos = poly.monoms()
            coeff = float(poly.coeffs()[0])
            for monom in monomos:  # Clasifica por grado
                suma_deg = sum(monom)
                if suma_deg == 0:
                    const += coeff
                elif suma_deg == 1:  # Actualiza vector lineal
                    idx = monom.index(1)
                    c_vec[idx] += coeff
                elif suma_deg == 2:  # Actualiza matriz Q
                    if 2 in monom:
                        idx = monom.index(2)
                        Q[idx, idx] += 2 * coeff
                    else:
                        idxs = [i for i, m in enumerate(monom) if m == 1]
                        i, j = idxs
                        Q[i, j] += coeff
                        Q[j, i] += coeff  # Simetriza
                else:
                    raise ValueError("Función no es cuadrática (grado > 2).")

        log("Matriz Q estimada:")
        log(formatear_matriz(Q))
        log("Vector c estimado:")
        log(formatear_vector(c_vec))

        eigs = np.linalg.eigvals(Q)
        log(f"Valores propios Q: {formatear_vector(eigs)}")
        convex = np.all(eigs >= -1e-10)
        log(f"Convexidad (Q semidef positiva): {convex}")

        has_constraints = bool(igualdades or desigualdades)
        if has_constraints and sco is not None:
            log("Hay restricciones: intentando resolver numéricamente con SLSQP.")

            def f_obj(x):  # Función objetivo QP: 1/2 x^TQx + c^T x + const
                return 0.5 * np.dot(x, Q.dot(x)) + np.dot(c_vec, x) + const

            def f_grad(x):  # Gradiente: Qx + c
                return Q.dot(x) + c_vec

            constraints = []
            if igualdades:  # Convierte igualdades a forma Ax=b
                try:
                    A_sym, b_sym = linear_eq_to_matrix(
                        [eq.lhs - eq.rhs for eq in igualdades], lista_vars
                    )  # Extrae matriz y vector
                    A = np.array([[float(N(A_sym[i, j]))
                                   for j in range(A_sym.shape[1])]
                                  for i in range(A_sym.shape[0])], dtype=float)  # Convierte a numpy
                    b = np.array([float(N(-b_sym[i]))
                                  for i in range(b_sym.shape[0])], dtype=float)  # Vector b
                    log("Matriz A (igualdades) extraída con linear_eq_to_matrix:")
                    log(formatear_matriz(A))
                    log("Vector b:")
                    log(formatear_vector(b))
                    for i in range(len(b)):  # Crea funciones de restricción eq
                        def eq_constraint(x, i=i, A=A, b=b):
                            return np.dot(A[i], x) - b[i]
                        constraints.append({'type': 'eq', 'fun': eq_constraint})
                except Exception:
                    log("Advertencia: linear_eq_to_matrix falló, usando método original para igualdades.")
                    for eq in igualdades:
                        expr_l = expand(eq.lhs - eq.rhs)
                        fn = crear_funcion_numpy(expr_l, lista_vars)
                        constraints.append({'type': 'eq', 'fun': lambda x, fn=fn: fn(x)})

            for r in desigualdades:
                try:
                    g = self._parsear_inecuacion(r)
                    fn = crear_funcion_numpy(g, lista_vars)
                    constraints.append({'type': 'ineq', 'fun': lambda x, fn=fn: fn(x)})
                except Exception:
                    log(f"No se pudo convertir desigualdad a numérico: {r}")

            init_text = parametros_grad.get("init", "").strip()
            try:
                init_vals = [float(v.strip()) for v in init_text.split(",")
                             if v.strip() != '']  # Parseo de x0
            except Exception:
                init_vals = []
            if len(init_vals) < n:
                init_vals = init_vals + [0.0] * (n - len(init_vals))
            x0 = np.array(init_vals[:n], dtype=float)

            res = None  # Resultado de optimización
            try:
                res = sco.minimize(
                    f_obj, x0, jac=f_grad, constraints=constraints, method='SLSQP',
                    options={'maxiter': int(a_flotante_seguro(
                        parametros_grad.get("max_iter", 1000), 1000))}
                )
            except Exception as e:
                log(f"SLSQP falló al ejecutar: {e}")

            if res is not None and res.success:  # Si éxito, muestra solución y termina
                log(f"Solución SLSQP: x = {formatear_vector(res.x)}, fun = {res.fun:.12g}")
                return
            else:
                log(f"SLSQP no encontró solución óptima: {getattr(res,'message', 'no result')}. Se intenta fallback.")
        elif has_constraints and sco is None:
            log("Hay restricciones pero scipy no está disponible: no se puede usar SLSQP. Se intentará solución aproximada (ignorando desigualdades).")

        try:  # Intenta resolver Qx=-c analíticamente
            xsol = np.linalg.solve(Q, -c_vec)  # Solución exacta si Q invertible
            log(f"Solución Qx = -c: x = {formatear_vector(xsol)}")
            subs = {lista_vars[i]: xsol[i] for i in range(n)}  # Sustituciones para evaluar f
            fval = evaluar_numerico_seguro(f, subs)  # Evalúa f en la solución
            if fval is None:
                log(f"f (no numérico) en x={formatear_vector(xsol)}: {texto_ecuacion(f.subs(subs))}")
            else:
                log(f"f = {fval:.12g}")
        except np.linalg.LinAlgError:  # Q singular: usa mínimos cuadrados
            log("Q singular: uso pseudo-inversa (mínimos cuadrados).")
            xsol, *_ = np.linalg.lstsq(Q, -c_vec, rcond=None)
            log(f"Solución aproximada: x = {formatear_vector(xsol)}")
            try:
                subs = {lista_vars[i]: xsol[i] for i in range(n)}
                fval = evaluar_numerico_seguro(f, subs)
                if fval is None:
                    log(f"f (no numérico) en x={formatear_vector(xsol)}: {texto_ecuacion(f.subs(subs))}")
                else:
                    log(f"f ≈ {fval:.12g}")
            except Exception:
                pass
