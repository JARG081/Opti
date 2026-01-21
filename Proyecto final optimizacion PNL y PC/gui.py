import tkinter as tk
from tkinter import ttk, messagebox
from sympy import Symbol

from optimizador import Optimizador


class OptimizadorGUI:
    def __init__(self, raiz):
        self.raiz = raiz
        self.raiz.title("Proyecto Final - Programación No Lineal")
        self.opt = Optimizador()
        self._crear_widgets()

    def _crear_widgets(self):
        ttk.Label(self.raiz, text="Función objetivo (ej: x**2 + y**2 + z):").grid(row=0, column=0, sticky="w")
        self.entrada_funcion = tk.Entry(self.raiz, width=70)
        self.entrada_funcion.insert(0, "x**2 + y**2")
        self.entrada_funcion.grid(row=0, column=1, columnspan=3, sticky="ew")

        ttk.Label(self.raiz, text="Restricciones (una por línea; usar '=' o '<=' o '>=')").grid(row=1, column=0, sticky="nw")
        self.texto_restricciones = tk.Text(self.raiz, width=70, height=6)
        self.texto_restricciones.insert("1.0", "x + y = 1")
        self.texto_restricciones.grid(row=1, column=1, columnspan=3, sticky="ew")

        ttk.Label(self.raiz, text="Usar sugerencia automática:").grid(row=2, column=0, sticky="w")
        self.var_usar_sugerido = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.raiz, variable=self.var_usar_sugerido).grid(row=2, column=1, sticky="w")

        ttk.Label(self.raiz, text="Método manual (si no usa sugerido):").grid(row=3, column=0, sticky="w")
        self.var_metodo_manual = tk.StringVar(value="Auto")
        opciones = [
            "Auto",
            "Sin restricciones",
            "Hessiana",
            "Lagrange (igualdad)",
            "KKT (desigualdad)",
            "Gradiente (iterativo)",
            "Programación Cuadrática"
        ]
        ttk.OptionMenu(self.raiz, self.var_metodo_manual, opciones[0], *opciones)\
            .grid(row=3, column=1, columnspan=2, sticky="ew")

        marco_grad = ttk.LabelFrame(self.raiz, text="Parámetros Gradiente")
        marco_grad.grid(row=4, column=0, columnspan=5, sticky="ew", padx=2, pady=4)
        ttk.Label(marco_grad, text="Alpha inicial:").grid(row=0, column=0, sticky="w")
        self.entrada_alpha = tk.Entry(marco_grad, width=8)
        self.entrada_alpha.insert(0, "1.0")
        self.entrada_alpha.grid(row=0, column=1)
        ttk.Label(marco_grad, text="Max iter:").grid(row=0, column=2, sticky="w")
        self.entrada_maxiter = tk.Entry(marco_grad, width=8)
        self.entrada_maxiter.insert(0, "1000")
        self.entrada_maxiter.grid(row=0, column=3)
        ttk.Label(marco_grad, text="Tol grad:").grid(row=0, column=4, sticky="w")
        self.entrada_tol = tk.Entry(marco_grad, width=10)
        self.entrada_tol.insert(0, "1e-6")
        self.entrada_tol.grid(row=0, column=5)
        ttk.Label(marco_grad, text="Punto inicio (coma sep):").grid(row=0, column=6, sticky="w")
        self.entrada_init = tk.Entry(marco_grad, width=20)
        self.entrada_init.insert(0, "0.0,0.0")
        self.entrada_init.grid(row=0, column=7)

        ttk.Button(self.raiz, text="Resolver", command=self.resolver).grid(row=5, column=0, pady=6)
        ttk.Button(self.raiz, text="Ej. Lagrange", command=self.ej_lagrange).grid(row=5, column=1)
        ttk.Button(self.raiz, text="Ej. KKT", command=self.ej_kkt).grid(row=5, column=2)
        ttk.Button(self.raiz, text="Ej. QP", command=self.ej_qp).grid(row=5, column=3)
        ttk.Button(self.raiz, text="Ej. Gradiente", command=self.ej_gradiente).grid(row=5, column=4)

        self.panel_resultado = tk.Text(self.raiz, width=120, height=30)
        self.panel_resultado.grid(row=6, column=0, columnspan=5, padx=5, pady=5, sticky="nsew")

        self.raiz.columnconfigure(1, weight=1)
        self.raiz.rowconfigure(6, weight=1)

    def _leer_entradas(self):
        func = self.entrada_funcion.get().strip()
        restr = self.texto_restricciones.get("1.0", tk.END).strip()
        lineas = [r.strip() for r in restr.splitlines() if r.strip()]
        return func, lineas

    def _mostrar_error(self, titulo, msg):
        messagebox.showerror(titulo, msg)

    # ---------- Acción Resolver ----------
    def resolver(self):
        self.panel_resultado.delete("1.0", tk.END)

        def log(mensaje=""):
            self.panel_resultado.insert(tk.END, mensaje + "\n")

        texto_func, lineas_restr = self._leer_entradas()
        parametros_grad = {
            "alpha": self.entrada_alpha.get(),
            "max_iter": self.entrada_maxiter.get(),
            "tol": self.entrada_tol.get(),
            "init": self.entrada_init.get()
        }
        metodo_manual = self.var_metodo_manual.get()
        usar_sugerido = self.var_usar_sugerido.get()

        self.opt.resolver(
            texto_func=texto_func,
            lineas_restr=lineas_restr,
            parametros_grad=parametros_grad,
            metodo_manual=metodo_manual,
            usar_sugerido=usar_sugerido,
            log=log,
            mostrar_error=self._mostrar_error
        )

    # ---------- Ejemplos ----------
    def ej_lagrange(self):
        self.entrada_funcion.delete(0, tk.END)
        self.entrada_funcion.insert(0, "x**2 + y**2")
        self.texto_restricciones.delete("1.0", tk.END)
        self.texto_restricciones.insert("1.0", "x + y = 1")
        self.var_metodo_manual.set("Auto")
        self.var_usar_sugerido.set(True)

    def ej_kkt(self):
        self.entrada_funcion.delete(0, tk.END)
        self.entrada_funcion.insert(0, "x**2 + y**2")
        self.texto_restricciones.delete("1.0", tk.END)
        self.texto_restricciones.insert("1.0", "x + y >= 4\nx >= 0\ny >= 0")
        self.var_metodo_manual.set("Auto")
        self.var_usar_sugerido.set(True)

    def ej_qp(self):
        self.entrada_funcion.delete(0, tk.END)
        self.entrada_funcion.insert(0, "3*x**2 + 2*x*y + 4*y**2 + 5*x - 6*y")
        self.texto_restricciones.delete("1.0", tk.END)
        self.texto_restricciones.insert("1.0", "x + 2*y = 4")
        self.var_metodo_manual.set("Programación Cuadrática")
        self.var_usar_sugerido.set(False)

    def ej_gradiente(self):
        self.entrada_funcion.delete(0, tk.END)
        self.entrada_funcion.insert(0, "x**2 + y**2 + sin(x)")
        self.texto_restricciones.delete("1.0", tk.END)
        self.texto_restricciones.insert("1.0", "")
        self.var_metodo_manual.set("Gradiente (iterativo)")
        self.var_usar_sugerido.set(False)
        self.entrada_init.delete(0, tk.END)
        self.entrada_init.insert(0, "0.5,0.5")
