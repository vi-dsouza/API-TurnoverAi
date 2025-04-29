import psycopg2 
import os 
from dotenv import load_dotenv
from psycopg2 import Error

load_dotenv()

password = os.getenv('password')

#def para fazer a conexão
def conecta():
    try:
        conn = psycopg2.connect(
            user='postgres',
            password=password,
            host='localhost',
            port='5432',
            database='funcionarios'
        )
        print("Conectado no postgres com sucesso!")

        return conn
    except Error as e:
        print(f'Ocorreu um erro ao tentar conectar no banco de dados {e}')

def encerra_conexao(conn):
    if conn:
        conn.close()
    print('Conexão encerrada com o banco de dados!')