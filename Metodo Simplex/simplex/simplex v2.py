import numpy as np
import tkinter as tk
from tkinter import Entry, Label, Button, StringVar

# Crea una matriz de ceros con las dimensiones dadas
def crear_matriz(filas, columnas):
    return [[0] * columnas for _ in range(filas)]

# Imprime la matriz en la interfaz con 2 decimales
def imprimir_matriz(matriz):
    texto = "\n".join(["\t".join(map(lambda x: f"{x:.2f}", fila)) for fila in matriz])
    steps_text.set(steps_text.get() + "\n" + texto + "\n\n")

# Encuentra la columna pivote en la última fila (Z)
def encontrar_columpiv(matriz, num_filas, num_colum):
    num_pivoteZ = 0
    colum_pivote = -1
    for j in range(num_colum - 1):
        if matriz[num_filas - 1][j] < num_pivoteZ:
            num_pivoteZ = matriz[num_filas - 1][j]
            colum_pivote = j
    return colum_pivote

# Encuentra el elemento pivote dividiendo el lado derecho entre la columna pivote
def encontrar_elemento_pivote(matriz, colum_pivote, num_filas, num_colum):
    fila_pivot = -1
    num_menor = float('inf')
    for i in range(num_filas - 1):
        if matriz[i][colum_pivote] > 0:
            razon = matriz[i][num_colum - 1] / matriz[i][colum_pivote]
            if razon < num_menor:
                num_menor = razon
                fila_pivot = i
    return fila_pivot, matriz[fila_pivot][colum_pivote]

# Ejecuta el método simplex paso a paso mostrando cada iteración
def simplex():
    global matriz_1, matriz_2, num_filas, num_colum
    salidaaux = 1
    while salidaaux == 1:
        imprimir_matriz(matriz_1)
        colum_pivote = encontrar_columpiv(matriz_1, num_filas, num_colum)
        fila_pivot, elemento_pivote = encontrar_elemento_pivote(matriz_1, colum_pivote, num_filas, num_colum)

        for j in range(num_colum):
            matriz_2[fila_pivot][j] = matriz_1[fila_pivot][j] / elemento_pivote

        for i in range(num_filas):
            if i != fila_pivot:
                factor = matriz_1[i][colum_pivote]
                for j in range(num_colum):
                    matriz_2[i][j] = matriz_1[i][j] - factor * matriz_2[fila_pivot][j]

        matriz_1 = [row[:] for row in matriz_2]
        salidaaux = any(matriz_1[num_filas - 1][j] < 0 for j in range(num_colum - 1))

    resultado_final = f"Solución óptima encontrada: Z = {matriz_1[num_filas - 1][-1]:.2f}"
    steps_text.set(steps_text.get() + "\n" + resultado_final + "\n")

# Obtiene los datos de la interfaz y resuelve el método simplex
def resolver_simplex():
    global matriz_1, matriz_2, num_filas, num_colum
    steps_text.set("")
    try:
        numero_varZ = int(entry_vars.get())
        numero_inec = int(entry_ineqs.get())
        num_filas = numero_inec + 1
        num_colum = numero_inec + numero_varZ + 2

        matriz_1 = crear_matriz(num_filas, num_colum)
        matriz_2 = crear_matriz(num_filas, num_colum)

        coef_z = list(map(int, entry_z.get().split()))
        for j in range(numero_varZ):
            matriz_1[num_filas - 1][j + 1] = -coef_z[j]

        for i in range(numero_inec):
            coef_ineq = list(map(int, restricciones_entries[i].get().split()))
            for j in range(numero_varZ):
                matriz_1[i][j + 1] = coef_ineq[j]
            matriz_1[i][num_colum - 1] = coef_ineq[-1]
            matriz_1[i][numero_varZ + 1 + i] = 1

        simplex()
        imprimir_matriz(matriz_1)
    except Exception as e:
        steps_text.set(f"Error en los datos: {e}")

# Interfaz gráfica con Tkinter
ventana = tk.Tk()
ventana.title("Método Simplex")

Label(ventana, text="Número de variables en Z:").pack()
entry_vars = Entry(ventana)
entry_vars.pack()

Label(ventana, text="Número de restricciones:").pack()
entry_ineqs = Entry(ventana)
entry_ineqs.pack()

Label(ventana, text="Función objetivo (coeficientes separados por espacio):").pack()
entry_z = Entry(ventana, width=40)
entry_z.pack()

restricciones_entries = []
Label(ventana, text="Restricciones (coeficientes separados por espacio):").pack()
for _ in range(5):
    e = Entry(ventana, width=40)
    e.pack()
    restricciones_entries.append(e)

Button(ventana, text="Resolver", command=resolver_simplex, bg='blue', fg='white').pack()
steps_text = StringVar()
steps_label = Label(ventana, textvariable=steps_text, justify=tk.CENTER, font=("Arial", 10))
steps_label.pack()

ventana.mainloop()
