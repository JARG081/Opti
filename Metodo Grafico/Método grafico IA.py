import os
import re
import json
import requests
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from PIL import Image, ImageTk
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ===========================
# Configuración de API (no hardcodees tu API key)
# ===========================
# Usa la variable de entorno OPENAI_API_KEY
API_KEY = "#AQUI_VA_SU_API_KEY_"
OPENAI_URL = "https://api.openai.com/v1/responses"
MODEL = "gpt-4o-mini"

# ===========================
# Utilidades de parseo
# ===========================

def _parse_objetivo(obj_str):
    s = obj_str.strip().lower()
    s = s.replace(" ", "")
    s = s.replace("z=", "")
    s = s.replace("x1", "x").replace("x2", "y")
    s = re.sub(r"(\d)(x)", r"\1*\2", s)
    s = re.sub(r"(\d)(y)", r"\1*\2", s)
    m_x = re.search(r"([+-]?\d*\.?\d*)\*?x", s)
    m_y = re.search(r"([+-]?\d*\.?\d*)\*?y", s)
    def _num(m):
        if not m:
            return 0.0
        v = m.group(1)
        if v in ("", "+", "-"):
            v = v + "1"
        return float(v)
    a = _num(m_x)
    b = _num(m_y)
    return a, b


import re

def _parse_restriccion(linea: str):
    """
    Parsea una restricción lineal de la forma:
        Ax + By <= C
        Ax + By >= C
        Ax + By = C
    y también soporta expresiones con variables en ambos lados,
    por ejemplo 'y >= 4x', transformándolas a la forma estándar.
    Devuelve (A, B, C) con desigualdad normalizada como <=.
    """
    # Normalización básica
    s = linea.strip().lower()
    s = s.replace("≤", "<=").replace("≥", ">=")
    s = s.replace("=<", "<=").replace("=>", ">=")
    s = s.replace(" ", "")

    # Normalizar nombres de variables
    s = s.replace("x1", "x").replace("x2", "y")

    # Insertar * entre número y variable (ej. 2x -> 2*x)
    s = re.sub(r"(\d)(x)", r"\1*\2", s)
    s = re.sub(r"(\d)(y)", r"\1*\2", s)

    # Buscar operador
    m = re.match(r"(.+?)(<=|>=|=)(.+)$", s)
    if not m:
        raise ValueError(f"Formato inválido de restricción: '{linea}'.")
    izq, op, der = m.group(1), m.group(2), m.group(3)

    # --- función para extraer coeficientes de x,y en una expresión ---
    def coeficientes(expr: str):
        expr = expr.replace("-", "+-")  # uniformizar
        terms = expr.split("+")
        A = B = C = 0.0
        for t in terms:
            if not t.strip():
                continue
            if "x" in t:
                v = t.replace("*x", "").replace("x", "")
                if v in ("", "+", "-"):
                    v += "1"
                A += float(v)
            elif "y" in t:
                v = t.replace("*y", "").replace("y", "")
                if v in ("", "+", "-"):
                    v += "1"
                B += float(v)
            else:
                C += float(t)
        return A, B, C

    # Parsear ambos lados
    A1, B1, C1 = coeficientes(izq)
    A2, B2, C2 = coeficientes(der)

    # Mover todo al lado izquierdo
    A = A1 - A2
    B = B1 - B2
    C = C2 - C1   # importante: el lado derecho pasa restando

    # Normalizar >= en <=
    if op == ">=":
        A, B, C = -A, -B, -C

    return A, B, C


# ===========================
# Núcleo: resolver y graficar (ahora con mínimos x>=Lx, y>=Ly)
# ===========================

def resolver_y_graficar(obj_str, tipo_str, r1_str, r2_str, canvas_widget, resultado_var, min_x=0.0, min_y=0.0):
    try:
        a, b = _parse_objetivo(obj_str)
        A1, B1, C1 = _parse_restriccion(r1_str)
        A2, B2, C2 = _parse_restriccion(r2_str)
    except Exception as e:
        messagebox.showerror("Error de formato", str(e))
        return

    # Candidatos de intersección
    candidatos = set()

    # Siempre incluir el punto de mínimos
    candidatos.add((float(min_x), float(min_y)))

    def inter_con_xmin(A, B, C, xmin):
        if abs(B) < 1e-12:
            return []
        y = (C - A * xmin) / B
        return [(xmin, y)]

    def inter_con_ymin(A, B, C, ymin):
        if abs(A) < 1e-12:
            return []
        x = (C - B * ymin) / A
        return [(x, ymin)]

    # Intersecciones con las rectas x=min_x e y=min_y
    for A, B, C in [(A1, B1, C1), (A2, B2, C2)]:
        for p in inter_con_xmin(A, B, C, min_x):
            candidatos.add((float(p[0]), float(p[1])))
        for p in inter_con_ymin(A, B, C, min_y):
            candidatos.add((float(p[0]), float(p[1])))

    # Intersección entre restricciones
    det = A1 * B2 - A2 * B1
    if abs(det) > 1e-12:
        x_int = (C1 * B2 - C2 * B1) / det
        y_int = (A1 * C2 - A2 * C1) / det
        candidatos.add((float(x_int), float(y_int)))

    # También considerar ejes si los mínimos son 0
    def inter_ejes(A, B, C):
        pts = []
        if abs(A) > 1e-12:
            x = C / A
            pts.append((x, 0.0))
        if abs(B) > 1e-12:
            y = C / B
            pts.append((0.0, y))
        return pts

    if min_x <= 1e-12 and min_y <= 1e-12:
        for A, B, C in [(A1, B1, C1), (A2, B2, C2)]:
            for p in inter_ejes(A, B, C):
                candidatos.add((float(p[0]), float(p[1])))

    # Factibilidad con mínimos
    def factible(x, y):
        tol = 1e-9
        return (
            A1 * x + B1 * y <= C1 + tol and
            A2 * x + B2 * y <= C2 + tol and
            x >= min_x - tol and y >= min_y - tol
        )

    vertices = [(x, y) for (x, y) in candidatos if np.isfinite(x) and np.isfinite(y) and factible(x, y)]
    if not vertices:
        messagebox.showwarning("Sin solución", "No hay región factible con las restricciones dadas y los mínimos especificados.")
        return

    # Evaluar objetivo
    tipo = tipo_str.strip().lower()
    def val(x, y):
        return a * x + b * y

    mejor = None
    mejor_p = None
    for (x, y) in vertices:
        v = val(x, y)
        if mejor is None or (tipo.startswith("max") and v > mejor) or (tipo.startswith("min") and v < mejor):
            mejor = v
            mejor_p = (x, y)

    _graficar(vertices, (A1, B1, C1), (A2, B2, C2), mejor_p, canvas_widget, min_x=min_x, min_y=min_y)

    x_opt, y_opt = mejor_p
    resultado_var.set(f"Solución óptima: (x, y) = ({x_opt:.2f}, {y_opt:.2f})  |  Z* = {mejor:.2f}  ({'Max' if tipo.startswith('max') else 'Min'})")


# ===========================
# Gráfica
# ===========================

def _ordenar_poligono(pts):
    # Ordenar puntos por ángulo alrededor del centroide para rellenar el polígono correctamente
    pts = list(set((float(x), float(y)) for x, y in pts))
    cx = sum(x for x, _ in pts) / len(pts)
    cy = sum(y for _, y in pts) / len(pts)
    pts.sort(key=lambda p: np.arctan2(p[1] - cy, p[0] - cx))
    return pts


def _graficar(vertices, r1, r2, punto_optimo, canvas_widget, min_x=0.0, min_y=0.0):
    verts = _ordenar_poligono(vertices)
    x_v, y_v = zip(*verts)

    plt.clf()
    plt.fill(x_v, y_v, alpha=0.2, label="Región factible")

    def dibujar_recta(A, B, C, etiqueta):
        xs_max = max(max(x_v + (punto_optimo[0],) if punto_optimo else x_v), 1) * 1.2
        xs = np.linspace(min_x, xs_max, 400)
        ys = []
        for x in xs:
            if abs(B) < 1e-12:
                ys.append(np.nan)
            else:
                ys.append((C - A * x) / B)
        plt.plot(xs, ys, label=etiqueta)

    dibujar_recta(*r1, etiqueta="Restricción 1")
    dibujar_recta(*r2, etiqueta="Restricción 2")

    # Líneas de mínimos visuales
    if min_x > 0:
        plt.axvline(min_x, linestyle=":", linewidth=0.8)
    if min_y > 0:
        plt.axhline(min_y, linestyle=":", linewidth=0.8)

    if punto_optimo:
        plt.scatter([punto_optimo[0]], [punto_optimo[1]], s=60, marker="o", label="Óptimo")

    maxx = max([p[0] for p in vertices] + ([punto_optimo[0]] if punto_optimo else [0])) * 1.15 + 1e-9
    maxy = max([p[1] for p in vertices] + ([punto_optimo[1]] if punto_optimo else [0])) * 1.15 + 1e-9
    maxx = max(maxx, min_x + 1)
    maxy = max(maxy, min_y + 1)

    plt.xlim(min_x, maxx)
    plt.ylim(min_y, maxy)
    plt.axhline(0, linewidth=0.7, color="black")
    plt.axvline(0, linewidth=0.7, color="black")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.title("Método gráfico (2 variables)")

    out = "plano.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    img = Image.open(out)
    photo = ImageTk.PhotoImage(img)
    canvas_widget.config(width=img.width, height=img.height)
    canvas_widget.image = photo
    canvas_widget.create_image(0, 0, anchor="nw", image=photo)


# ===========================
# Llamada a la API y post-proceso
# ===========================

def call_openai_raw(prompt):
    if not API_KEY or API_KEY.strip() == "":
        raise RuntimeError("Falta OPENAI_API_KEY en el entorno. Configura tu API key en variables de entorno.")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }
    payload = {
        "model": MODEL,
        "input": prompt,
    }
    r = requests.post(OPENAI_URL, headers=headers, json=payload, timeout=30)
    return r


def intentar_extraer_json_de_texto(texto):
    if not texto or not isinstance(texto, str):
        return None
    possible_indexes = [m.start() for m in re.finditer(r"\{", texto)]
    if not possible_indexes:
        return None
    for start in possible_indexes:
        for end_match in reversed(list(re.finditer(r"\}", texto))):
            end = end_match.end()
            sub = texto[start:end]
            try:
                data = json.loads(sub)
                return data
            except Exception:
                continue
    try:
        return json.loads(texto)
    except Exception:
        return None


def procesar_respuesta_api_en_gui(response_obj, raw_text_widget, obj_entry, tipo_var, restr_entries, canvas_widget, resultado_var, min_x, min_y):
    raw_text_widget.config(state="normal")
    raw_text_widget.delete("1.0", tk.END)
    try:
        raw = response_obj.text
        raw_text_widget.insert(tk.END, raw)
    except Exception as e:
        raw_text_widget.insert(tk.END, f"(no se pudo obtener texto crudo) {e}")
    raw_text_widget.config(state="disabled")

    parsed = None
    try:
        j = response_obj.json()
        parsed_text = None
        try:
            parsed_text = j.get("output", [])[0].get("content", [])[0].get("text")
        except Exception:
            parsed_text = None
        if not parsed_text:
            parsed_text = j.get("output_text")
        if not parsed_text:
            try:
                parsed_text = j["choices"][0]["message"]["content"]
            except Exception:
                parsed_text = None
        if parsed_text:
            parsed = intentar_extraer_json_de_texto(parsed_text)
        if parsed is None:
            parsed = intentar_extraer_json_de_texto(raw)
    except Exception:
        parsed = intentar_extraer_json_de_texto(response_obj.text)

    if parsed:
        # normalizar claves
        tipo = None
        for k in parsed.keys():
            if k.lower().startswith("t"):
                tipo = parsed[k]
                break
        objetivo = None
        for k in parsed.keys():
            if "obj" in k.lower():
                objetivo = parsed[k]
                break
        restricciones = None
        for k in parsed.keys():
            if "restr" in k.lower():
                restricciones = parsed[k]
                break

        if not (tipo and objetivo and restricciones):
            messagebox.showwarning("JSON parcial", "Se encontró JSON pero faltan campos esperados (tipo/objetivo/restricciones). Revisa la salida cruda.")
            return
        if not isinstance(restricciones, list):
            messagebox.showwarning("Formato restricciones", "El campo 'restricciones' no es una lista. Revisa la salida cruda.")
            return
        if len(restricciones) < 2:
            messagebox.showwarning("Restricciones insuficientes", "La IA devolvió menos de 2 restricciones.")
            return

        # Setear en GUI (solo 2 restricciones)
        tipo_norm = "Maximizar" if str(tipo).strip().lower().startswith("max") else "Minimizar"
        tipo_var.set(tipo_norm)
        obj_entry.delete(0, tk.END)
        obj_entry.insert(0, objetivo)
        for i in range(2):
            restr_entries[i].delete(0, tk.END)
            restr_entries[i].insert(0, restricciones[i])

        # Sanity check: advertir si alguna restricción omite x o y
        try:
            Ax, Ay, _ = _parse_restriccion(restricciones[0])
            Bx, By, _ = _parse_restriccion(restricciones[1])
            if (abs(Ax) < 1e-12 or abs(Ay) < 1e-12) or (abs(Bx) < 1e-12 or abs(By) < 1e-12):
                messagebox.showinfo(
                    "Verifica restricciones",
                    "Alguna restricción no incluye x o y (coeficiente 0). Si ambos lotes consumen ambos recursos, revisa que la segunda restricción no deba ser, por ejemplo, 'x + y <= ...'."
                )
        except Exception:
            pass

        # Resolver y graficar con mínimos
        resolver_y_graficar(
            obj_entry.get(), tipo_var.get(), restr_entries[0].get(), restr_entries[1].get(),
            canvas_widget, resultado_var, min_x=min_x, min_y=min_y
        )
    else:
        messagebox.showinfo(
            "Salida cruda",
            "Se ha mostrado la respuesta cruda de la API. Si contiene JSON, el programa intentó extraerlo.\n\nSi no se llenaron los campos, copia el JSON mostrado y pégalo en el campo Manual o edítalo para que la IA devuelva JSON limpio."
        )


# ===========================
# Interfaz Tk
# ===========================

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Método Gráfico con IA - GUI (genérico)")
        self.geometry("1220x760")

        main = ttk.Frame(self)
        main.pack(fill="both", expand=True, padx=8, pady=8)

        left = ttk.Frame(main)
        left.pack(side="left", fill="both", expand=True)

        self.canvas = tk.Canvas(left, width=680, height=620, bg="white")
        self.canvas.pack(fill="both", expand=True)

        right = ttk.Frame(main, width=520)
        right.pack(side="right", fill="y")

        nb = ttk.Notebook(right)
        nb.pack(fill="both", expand=True)

        # ---- Tab Manual ----
        tab_manual = ttk.Frame(nb)
        nb.add(tab_manual, text="Modo Manual")

        ttk.Label(tab_manual, text="Tipo de problema:").pack(pady=(10, 0))
        self.tipo_var = tk.StringVar(value="Maximizar")
        frm_tipo = ttk.Frame(tab_manual)
        frm_tipo.pack()
        ttk.Radiobutton(frm_tipo, text="Maximizar", variable=self.tipo_var, value="Maximizar").pack(side="left", padx=4)
        ttk.Radiobutton(frm_tipo, text="Minimizar", variable=self.tipo_var, value="Minimizar").pack(side="left", padx=4)

        ttk.Label(tab_manual, text="Función objetivo (ej: Z = 5x + 3y):").pack(pady=(8, 0))
        self.obj_entry = ttk.Entry(tab_manual, width=48)
        self.obj_entry.pack()

        ttk.Label(tab_manual, text="Restricciones (exactamente 2):").pack(pady=(8, 0))
        self.restr_entries = []
        for i in range(2):
            e = ttk.Entry(tab_manual, width=48)
            e.pack(pady=3)
            self.restr_entries.append(e)

        # Campos para mínimos x>=Lx, y>=Ly
        frm_min = ttk.Frame(tab_manual)
        frm_min.pack(pady=(8, 0))
        ttk.Label(frm_min, text="Mínimos (si aplica):  x ≥").pack(side="left")
        self.min_x_entry = ttk.Entry(frm_min, width=6)
        self.min_x_entry.insert(0, "0")
        self.min_x_entry.pack(side="left", padx=(4, 12))
        ttk.Label(frm_min, text="y ≥").pack(side="left")
        self.min_y_entry = ttk.Entry(frm_min, width=6)
        self.min_y_entry.insert(0, "0")
        self.min_y_entry.pack(side="left", padx=4)

        frm_buttons = ttk.Frame(tab_manual)
        frm_buttons.pack(pady=8)
        ttk.Button(
            frm_buttons,
            text="Calcular y Graficar",
            command=lambda: resolver_y_graficar(
                self.obj_entry.get(), self.tipo_var.get(),
                self.restr_entries[0].get(), self.restr_entries[1].get(),
                self.canvas, self.resultado_var,
                min_x=float(self.min_x_entry.get() or 0),
                min_y=float(self.min_y_entry.get() or 0),
            ),
        ).pack(side="left", padx=4)
        ttk.Button(frm_buttons, text="Cargar ejemplo", command=self._cargar_ejemplo).pack(side="left", padx=4)

        # ---- Tab Texto (IA) ----
        tab_texto = ttk.Frame(nb)
        nb.add(tab_texto, text="Modo Texto (IA)")

        ttk.Label(tab_texto, text="Pega aquí el enunciado (libro):").pack(pady=(8, 0))
        self.texto_widget = scrolledtext.ScrolledText(tab_texto, width=60, height=14, wrap=tk.WORD)
        self.texto_widget.pack(pady=6)

        btns = ttk.Frame(tab_texto)
        btns.pack()
        ttk.Button(btns, text="Enviar a OpenAI y Mostrar crudo + Extraer JSON", command=self._enviar_a_ia).pack(padx=4, pady=6)

        ttk.Label(tab_texto, text="Respuesta cruda de la API:").pack(pady=(6, 0))
        self.raw_text = scrolledtext.ScrolledText(tab_texto, width=60, height=10, wrap=tk.WORD, state="disabled")
        self.raw_text.pack(pady=4)

        # Resultado general
        self.resultado_var = tk.StringVar(value="")
        ttk.Label(right, textvariable=self.resultado_var, foreground="#0a6").pack(pady=10)

        # Hint/API note
        hint = (
            "Nota: la IA debe devolver JSON con campos: tipo, objetivo, restricciones (lista de 2 strings).\n"
            "Ejemplo: { \"tipo\": \"Maximizar\", \"objetivo\": \"Z = 5x + 3y\", \"restricciones\": [\"2x + y <= 8\", \"x + y <= 6\"] }\n"
            "Si el enunciado impone mínimos (p. ej. x≥20, y≥10), ingrésalos en los campos de mínimos."
        )
        ttk.Label(right, text=hint, wraplength=420, foreground="#666").pack(pady=8)

    def _cargar_ejemplo(self):
        # Ejemplo de las camisas-pantalones correcto
        self.tipo_var.set("Maximizar")
        self.obj_entry.delete(0, tk.END)
        self.obj_entry.insert(0, "Z = 30x + 50y")
        self.restr_entries[0].delete(0, tk.END)
        self.restr_entries[0].insert(0, "x + 3y <= 200")
        self.restr_entries[1].delete(0, tk.END)
        self.restr_entries[1].insert(0, "x + y <= 100")
        # Mínimos del enunciado: x>=20, y>=10
        self.min_x_entry.delete(0, tk.END)
        self.min_x_entry.insert(0, "20")
        self.min_y_entry.delete(0, tk.END)
        self.min_y_entry.insert(0, "10")

    def _enviar_a_ia(self):
        texto = self.texto_widget.get("1.0", tk.END).strip()
        if not texto:
            messagebox.showerror("Error", "Pega primero el enunciado.")
            return

        # Prompt robusto: fuerza a usar x e y y a construir restricciones por consumo de recursos
        prompt = f"""
            Eres un asistente experto en programación lineal de 2 variables.
            Lee el enunciado y devuelve ÚNICAMENTE un JSON con estas claves exactas:
            - tipo: "Maximizar" o "Minimizar"
            - objetivo: usa variables x e y (ej: "Z = 5x + 3y")
            - restricciones: lista con EXACTAMENTE 2 strings en forma algebraica (ej: "2x + y <= 8").

            Reglas:
            - Usa SIEMPRE variables x e y.
            - Si dos lotes consumen un mismo recurso (p. ej., ambos llevan 1 pantalón), la restricción de ese recurso es la SUMA de ambos (ej.: x + y <= total de pantalones), no solo una variable.
            - No agregues texto adicional ni explicaciones. Devuelve SOLO JSON válido.

            Enunciado:
            {texto}
            """
        try:
            resp = call_openai_raw(prompt)
        except Exception as e:
            messagebox.showerror("Error API", f"No se pudo conectar con OpenAI:\n{e}")
            return

        try:
            # Tomar mínimos actuales del UI para respetarlos al resolver
            min_x = float(self.min_x_entry.get() or 0)
            min_y = float(self.min_y_entry.get() or 0)
        except Exception:
            min_x, min_y = 0.0, 0.0

        try:
            procesar_respuesta_api_en_gui(
                resp, self.raw_text, self.obj_entry, self.tipo_var, self.restr_entries,
                self.canvas, self.resultado_var, min_x=min_x, min_y=min_y
            )
        except Exception as e:
            messagebox.showerror("Error", f"Error procesando respuesta:\n{e}")


# ===========================
# Ejecutar app
# ===========================
if __name__ == "__main__":
    app = App()
    app.mainloop()
