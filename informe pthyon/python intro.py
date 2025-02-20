import matplotlib.pyplot as plt
import pandas as pd

"""
entero1 = 10 #entero
print(entero1)
entero2 = 30 #entero
print(entero2)
decimal1 = 2.5 #float
print(decimal1)
string1 = "Esto es un string" #string
print(string1)

booleano1 = False #booleano
print(booleano1)
booleano2 = True #booleano
print(booleano2)
suma = entero1 + entero2
print(suma)
resta = entero2 - entero1
print(resta)
comparacion1 = (entero1==entero2)
print(comparacion1)

#condicionales
if(comparacion1):
    print("Los enteros son iguales")
else:
    print("Los enteros son diferentes")

if(entero1==0):
    print("El entero 1 es 0")
elif(entero1==2):
    print("El entero 1 es 2")
else:
    print("El entero es otro numero")


#ciclos

for i in range(5):
    print("Hey")
cont=1
while cont < 10:
    print(cont)
    cont+=1

#listas
lista = ["elem0","elem1","elem2"]
print(lista[0])
print(lista[2])
print(lista[1])
print(len(lista))
print("Se agrega un elemento a la lista")
lista = ["elem0","elem1","elem2","elem3"]
print(len(lista))

for i in range(len(lista)):
    print(lista[i])
"""
#graficos

eje_y = [0,2,4,6,8,10]
eje_x = [0,1,2,3,4,5]

plt.plot(eje_x,eje_y)
plt.xlabel("Eje X")
plt.ylabel("Eje Y")
plt.title("Grafica")
plt.show()


#datos importados
 
data = pd.read_csv("paises_capitales.csv")
print(data.head())


