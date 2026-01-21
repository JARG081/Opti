import tkinter as tk
from tkinter import messagebox, ttk
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class GrafoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Programa de Grafos")
        self.entries = []
        self.n = 0
        self.dirigido = tk.BooleanVar(value=True)
        self.setup_input()
        self.root.geometry("420x330")

    def setup_input(self):
        tk.Label(self.root, text="Número de vértices:").grid(row=0, column=0, padx=5, pady=5)
        self.vertex_entry = tk.Entry(self.root)
        self.vertex_entry.grid(row=0, column=1, padx=5, pady=5)

        # Tipo de grafo
        tk.Label(self.root, text="Tipo de grafo:").grid(row=1, column=0, padx=5, pady=5)
        tipo_menu = ttk.Combobox(self.root, values=["Dirigido", "No dirigido"], state="readonly")
        tipo_menu.current(0)
        tipo_menu.grid(row=1, column=1, padx=5, pady=5)
        tipo_menu.bind("<<ComboboxSelected>>", lambda e: self.dirigido.set(tipo_menu.get() == "Dirigido"))

        tk.Button(self.root, text="Crear matriz", command=self.crear_matriz).grid(row=0, column=2, padx=5, pady=5, rowspan=2)

    def crear_matriz(self):
        # Eliminar widgets anteriores excepto los de entrada
        for widget in self.root.winfo_children():
            if widget.grid_info()["row"] >= 3:
                widget.destroy()

        self.entries.clear()

        try:
            n = int(self.vertex_entry.get())
            if n <= 0:
                raise ValueError
            self.n = n
        except ValueError:
            messagebox.showerror("Error", "Ingrese un número entero válido mayor que 0.")
            return

        encabezado = tk.Label(self.root, text="Matriz de adyacencia (pesos de las aristas)", font=("Arial", 12, "bold"))
        encabezado.grid(row=3, column=0, columnspan=n + 2, pady=10)

        tk.Label(self.root, text="").grid(row=4, column=0, padx=10)
        for j in range(n):
            tk.Label(self.root, text=f"V{j}", width=5, anchor="center").grid(row=4, column=1 + j, padx=5, pady=2)

        for i in range(n):
            tk.Label(self.root, text=f"V{i}", width=5, anchor="w").grid(row=5 + i, column=0, padx=10, pady=2)
            row_entries = []
            for j in range(n):
                e = tk.Entry(self.root, width=6, justify="center")
                e.grid(row=5 + i, column=1 + j, padx=5, pady=5)
                row_entries.append(e)
            self.entries.append(row_entries)

        tk.Button(self.root, text="Generar grafo", command=self.generar_grafo).grid(row=5 + n, column=0, columnspan=n + 2, pady=10)

    def generar_grafo(self):
        matriz = []
        for i, row in enumerate(self.entries):
            fila = []
            for j, entry in enumerate(row):
                val = entry.get().strip()
                try:
                    peso = float(val) if val != "" else 0.0
                except ValueError:
                    messagebox.showerror("Error", f"Peso inválido en V{i} a V{j}")
                    return
                fila.append(peso)
            matriz.append(fila)

        # Validación: evitar pesos duplicados en dirección contraria cuando es dirigido
        if self.dirigido.get():
            for i in range(self.n):
                for j in range(self.n):
                    if i != j and matriz[i][j] != 0 and matriz[j][i] != 0:
                        if matriz[i][j] == matriz[j][i]:
                            messagebox.showerror(
                                "Error",
                                f"No puede tener el mismo peso en ambas direcciones entre V{i} y V{j}.\n"
                                f"V{i}→V{j} = {matriz[i][j]}, V{j}→V{i} = {matriz[j][i]}"
                            )
                            return

        vertices = [f"V{i}" for i in range(self.n)]
        aristas = []

        G = nx.DiGraph() if self.dirigido.get() else nx.Graph()

        for i in range(self.n):
            G.add_node(vertices[i])
            for j in range(self.n):
                peso = matriz[i][j]
                if peso != 0:
                    if not self.dirigido.get() and i > j:
                        continue
                    G.add_edge(vertices[i], vertices[j], weight=peso)
                    aristas.append((vertices[i], vertices[j], peso))

        self.mostrar_grafo_completo(G, vertices, aristas)

    def mostrar_grafo_completo(self, G, vertices, aristas):
        top = tk.Toplevel(self.root)
        tipo = "Dirigido" if self.dirigido.get() else "No Dirigido"
        top.title(f"Grafo {tipo}")
        top.geometry("800x600")

        frame_texto = ttk.Frame(top)
        frame_texto.pack(fill=tk.X, padx=10, pady=10)

        expresion = f"V = {{{', '.join(vertices)}}}\n"
        expresion += "E = {" + ", ".join([f"({u}, {v}, {w})" for u, v, w in aristas]) + "}"

        lbl_conjunto = ttk.Label(frame_texto, text="Representación matemática:", font=('Arial', 10, 'bold'))
        lbl_conjunto.pack(anchor=tk.W)

        texto_conjunto = tk.Text(frame_texto, height=4, wrap=tk.WORD)
        texto_conjunto.insert(tk.END, expresion)
        texto_conjunto.pack(fill=tk.X, padx=5, pady=5)
        texto_conjunto.config(state=tk.DISABLED)

        frame_grafico = ttk.Frame(top)
        frame_grafico.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        fig = plt.figure(figsize=(7, 5))
        pos = nx.spring_layout(G)

        nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=800)
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

        curved_edges = []
        straight_edges = list(G.edges())

        if self.dirigido.get():
            curved_edges = [edge for edge in G.edges() if (edge[1], edge[0]) in G.edges()]
            straight_edges = [edge for edge in G.edges() if edge not in curved_edges]

        if self.dirigido.get():
            nx.draw_networkx_edges(G, pos, edgelist=straight_edges, arrowstyle='-|>', arrowsize=15)
            nx.draw_networkx_edges(G, pos, edgelist=curved_edges,
                                   connectionstyle='arc3, rad=0.2',
                                   arrowstyle='-|>', arrowsize=15)
        else:
            nx.draw_networkx_edges(G, pos, edgelist=straight_edges)

        # Etiquetas de peso con flechas si son bidireccionales
        edge_labels = nx.get_edge_attributes(G, 'weight')
        if self.dirigido.get():
            for edge in list(edge_labels):
                u, v = edge
                if (v, u) in edge_labels:
                    if u < v:
                        edge_labels[(u, v)] = f"{edge_labels[(u, v)]} ↗"
                        edge_labels[(v, u)] = f"{edge_labels[(v, u)]} ↘"
                    else:
                        edge_labels[(u, v)] = f"{edge_labels[(u, v)]} ↘"
                        edge_labels[(v, u)] = f"{edge_labels[(v, u)]} ↗"

        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                     font_size=8, label_pos=0.6,
                                     bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

        plt.title(f"Grafo {('Dirigido' if self.dirigido.get() else 'No Dirigido')}", fontsize=10)
        plt.axis('off')

        canvas = FigureCanvasTkAgg(fig, master=frame_grafico)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


if __name__ == "__main__":
    root = tk.Tk()
    app = GrafoApp(root)
    root.mainloop()
