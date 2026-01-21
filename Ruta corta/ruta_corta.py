import tkinter as tk
from tkinter import messagebox
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class ShortestPathApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Algoritmos de Ruta Más Corta")
        self.adj_matrix = []
        self.nodes = 0
        # Layout: panel principal con área izquierda (controles + grafo) y lateral derecho (pasos)
        main_frame = tk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        left_frame = tk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=6)

        right_frame = tk.Frame(main_frame, width=360)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=6, pady=6)

        # Entrada de matriz (izquierda)
        tk.Label(left_frame, text="Matriz de adyacencia (usar 'inf' para infinito):").pack()
        self.text_input = tk.Text(left_frame, height=10, width=60)
        self.text_input.pack()

        tk.Button(left_frame, text="Cargar Matriz", command=self.load_matrix).pack(pady=5)
        tk.Button(left_frame, text="Ejemplo Dijkstra", command=self.load_example_dijkstra).pack(pady=2)
        tk.Button(left_frame, text="Ejemplo Bellman-Ford", command=self.load_example_bellman).pack(pady=2)

        # Cargar un ejemplo 4x4 por defecto (algunas entradas vacías = sin arista)
        example = (
            "0,3,inf,7\n"
            "3,0,1,\n"
            "inf,1,0,2\n"
            "7,,2,0"
        )
        self.text_input.insert("1.0", example)

        # Nodos origen y destino
        self.origin_var = tk.StringVar()
        self.dest_var = tk.StringVar()
        tk.Label(left_frame, text="Nodo Origen:").pack()
        tk.Entry(left_frame, textvariable=self.origin_var).pack()
        tk.Label(left_frame, text="Nodo Destino:").pack()
        tk.Entry(left_frame, textvariable=self.dest_var).pack()

        # Botones de algoritmo
        tk.Button(left_frame, text="Dijkstra", command=self.run_dijkstra).pack(pady=5)
        tk.Button(left_frame, text="Bellman-Ford", command=self.run_bellman_ford).pack(pady=5)

        # Resultado
        self.result_label = tk.Label(left_frame, text="", fg="blue")
        self.result_label.pack()

        # Área gráfica
        self.fig, self.ax = plt.subplots(figsize=(5, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=left_frame)
        self.canvas.get_tk_widget().pack()

        # Panel derecho: pasos del algoritmo
        tk.Label(right_frame, text="Pasos del algoritmo:").pack(anchor='nw')
        self.steps_text = tk.Text(right_frame, width=48, height=30, state=tk.NORMAL)
        self.steps_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        steps_scroll = tk.Scrollbar(right_frame, command=self.steps_text.yview)
        steps_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.steps_text['yscrollcommand'] = steps_scroll.set
        tk.Button(right_frame, text="Limpiar pasos", command=self.clear_steps).pack(pady=4)

        # Inicialmente vacío
        self.clear_steps()

    def load_matrix(self):
        raw = self.text_input.get("1.0", tk.END).strip().split("\n")
        try:
            matrix = []
            for row in raw:
                # Mantener separación por comas; celdas vacías significan sin arista
                tokens = [t.strip() for t in row.split(',')]
                parsed = []
                for t in tokens:
                    if t == '' or t.lower() in ('inf', 'none'):
                        parsed.append(float('inf'))
                    else:
                        parsed.append(float(t))
                matrix.append(parsed)

            self.adj_matrix = matrix
            self.nodes = len(self.adj_matrix)
            for row in self.adj_matrix:
                if len(row) != self.nodes:
                    raise ValueError("Matriz no es cuadrada")

            # Validación básica: comprobar valores
            for i in range(self.nodes):
                for j in range(self.nodes):
                    v = self.adj_matrix[i][j]
                    if not (isinstance(v, float) or isinstance(v, int)):
                        raise ValueError(f"Valor no numérico en ({i},{j})")

            messagebox.showinfo("Éxito", "Matriz cargada correctamente")
        except Exception as e:
            messagebox.showerror("Error", f"Matriz inválida: {e}")

    def load_example_dijkstra(self):
        """Carga una matriz 4x4 de ejemplo sin pesos negativos para Dijkstra."""
        example = (
            "0,3,inf,7\n"
            "3,0,1,inf\n"
            "inf,1,0,2\n"
            "7,inf,2,0"
        )
        self.text_input.delete("1.0", tk.END)
        self.text_input.insert("1.0", example)

    def load_example_bellman(self):
        """Carga una matriz 4x4 de ejemplo que incluye un peso negativo (sin ciclo negativo) para Bellman-Ford."""
        example = (
            "0,4,inf,5\n"
            "inf,0,3,inf\n"
            "inf,-1,0,2\n"
            "inf,inf,inf,0"
        )
        self.text_input.delete("1.0", tk.END)
        self.text_input.insert("1.0", example)

    # Registro de pasos
    def log_step(self, msg: str):
        try:
            self.steps_text.insert(tk.END, msg + "\n")
            self.steps_text.see(tk.END)
            # refrescar UI brevemente
            self.root.update_idletasks()
        except Exception:
            pass

    def clear_steps(self):
        try:
            self.steps_text.delete("1.0", tk.END)
        except Exception:
            pass

    def run_dijkstra(self):
        try:
            origin = int(self.origin_var.get())
            dest = int(self.dest_var.get())
            if origin < 0 or origin >= self.nodes or dest < 0 or dest >= self.nodes:
                raise ValueError("Origen o destino fuera de rango")

            # Dijkstra no soporta pesos negativos
            for i in range(self.nodes):
                for j in range(self.nodes):
                    w = self.adj_matrix[i][j]
                    if w != float('inf') and w < 0:
                        raise ValueError("Dijkstra no soporta aristas con peso negativo")
            # Preparar panel de pasos
            self.clear_steps()
            self.log_step(f"Ejecutando Dijkstra desde {origin} hasta {dest}")

            dist, path = self.dijkstra(origin, dest)
            self.show_result(dist, path, "Dijkstra")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def run_bellman_ford(self):
        try:
            origin = int(self.origin_var.get())
            dest = int(self.dest_var.get())
            if origin < 0 or origin >= self.nodes or dest < 0 or dest >= self.nodes:
                raise ValueError("Origen o destino fuera de rango")

            # Preparar panel de pasos
            self.clear_steps()
            self.log_step(f"Ejecutando Bellman-Ford desde {origin} hasta {dest}")

            dist, path = self.bellman_ford(origin, dest)
            self.show_result(dist, path, "Bellman-Ford")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def dijkstra(self, start, end):
        visited = [False] * self.nodes
        dist = [float('inf')] * self.nodes
        prev = [None] * self.nodes
        dist[start] = 0
        self.log_step(f"Inicializar: dist={dist}, prev={prev}")

        for _ in range(self.nodes):
            # Seleccionar el vértice no visitado con menor distancia
            u = None
            best = float('inf')
            for i in range(self.nodes):
                if not visited[i] and dist[i] < best:
                    best = dist[i]
                    u = i
            if u is None:
                break
            visited[u] = True
            self.log_step(f"Seleccionado u={u} con distancia {dist[u]}")
            for v, w in enumerate(self.adj_matrix[u]):
                if not visited[v] and w != float('inf'):
                    old = dist[v]
                    if dist[u] + w < dist[v]:
                        dist[v] = dist[u] + w
                        prev[v] = u
                        self.log_step(f"Relajar arista {u}->{v} (w={w}): dist[{v}] {old} -> {dist[v]}")

            self.log_step(f"Estado actual: dist={dist}, prev={prev}")

        # Reconstruir camino
        if dist[end] == float('inf'):
            self.log_step(f"No existe ruta desde {start} hasta {end}")
            return float('inf'), []

        path = []
        at = end
        while at is not None:
            path.insert(0, at)
            at = prev[at]
        self.log_step(f"Camino encontrado: {path} con costo {dist[end]}")
        return dist[end], path

    def bellman_ford(self, start, end):
        dist = [float('inf')] * self.nodes
        prev = [None] * self.nodes
        dist[start] = 0
        self.log_step(f"Inicializar: dist={dist}, prev={prev}")

        for k in range(self.nodes - 1):
            self.log_step(f"Iteración {k+1}/{self.nodes-1}")
            for u in range(self.nodes):
                for v in range(self.nodes):
                    w = self.adj_matrix[u][v]
                    if w != float('inf') and dist[u] != float('inf') and dist[u] + w < dist[v]:
                        old = dist[v]
                        dist[v] = dist[u] + w
                        prev[v] = u
                        self.log_step(f"Relajar arista {u}->{v} (w={w}): dist[{v}] {old} -> {dist[v]}")

            self.log_step(f"Estado tras iteración {k+1}: dist={dist}, prev={prev}")

        # Comprobar ciclo negativo
        for u in range(self.nodes):
            for v in range(self.nodes):
                w = self.adj_matrix[u][v]
                if w != float('inf') and dist[u] != float('inf') and dist[u] + w < dist[v]:
                    self.log_step("Ciclo negativo detectado")
                    raise ValueError("Ciclo negativo detectado")

        if dist[end] == float('inf'):
            self.log_step(f"No existe ruta desde {start} hasta {end}")
            return float('inf'), []

        path = []
        at = end
        while at is not None:
            path.insert(0, at)
            at = prev[at]
        self.log_step(f"Camino encontrado: {path} con costo {dist[end]}")
        return dist[end], path

    def show_result(self, distance, path, algo):
        if distance == float('inf'):
            self.result_label.config(text=f"{algo} - No existe ruta entre los nodos especificados.")
        else:
            self.result_label.config(text=f"{algo} - Costo: {distance}, Ruta: {' → '.join(map(str, path))}")
        self.plot_graph(path)

    def plot_graph(self, path):
        self.ax.clear()
        G = nx.DiGraph()
        labels = {}
        for i in range(self.nodes):
            for j in range(self.nodes):
                w = self.adj_matrix[i][j]
                if w != float('inf'):
                    G.add_edge(i, j, weight=w)
                    labels[(i, j)] = w

        pos = nx.spring_layout(G)
        nx.draw(G, pos, ax=self.ax, with_labels=True, node_color="lightblue", node_size=600)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, ax=self.ax)
        edges_in_path = list(zip(path, path[1:])) if path and len(path) > 1 else []
        if edges_in_path:
            nx.draw_networkx_edges(G, pos, edgelist=edges_in_path, edge_color="red", width=2, ax=self.ax)
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = ShortestPathApp(root)
    root.mainloop()
