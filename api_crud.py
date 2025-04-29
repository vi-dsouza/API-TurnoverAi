from database import conecta, encerra_conexao
from flask import Flask, jsonify, request

app = Flask(__name__)

#create
@app.route('/acoes/cadastro', methods=['POST'])
def insert_acoes():
    data = request.get_json()

    ticker = data.get('ticker')
    nome_empresa = data.get('nome_empresa')
    setor = data.get('setor')
    preco = data.get('preco')
    data_criacao = data.get('data_criacao')

    connection = conecta()
    cursor = connection.cursor()

    cmd_insert = "INSERT INTO acoes_b3 (ticker, nome_empresa, setor, preco, data_criacao) VALUES (%s, %s, %s, %s, %s);"
    values = (ticker, nome_empresa, setor, preco, data_criacao)

    try:
        cursor.execute(cmd_insert, values)
        connection.commit()
        print('Dados inseridos com sucesso!')
        return jsonify({'message': 'Dados inseridos com sucesso!'}), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        encerra_conexao(connection)

#read
@app.route('/acoes/consulta', methods=['GET'])
def seleciona():

    connection = conecta()
    cursor = connection.cursor()

    cmd_select = "SELECT ticker, nome_empresa, setor, preco, data_criacao FROM acoes_b3;"

    try:    
        cursor.execute(cmd_select)
        acoes = cursor.fetchall()

        dados = []

        for acao in acoes:
            dados.append({
                'ticker': acao[0],
                'nome_empresa': acao[1],
                'setor': acao[2],
                'preco': acao[3],
                'data_criacao': acao[4].isoformat() if acao[4] else None
            })
        
        return jsonify(dados)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        encerra_conexao(connection)

#update
@app.route('/acoes/atualiza/<int:id>', methods=['PUT']) 
def atualiza(id):
    dados = request.get_json()
    novo_preco = dados.get('preco')
    

    if novo_preco is None:
        return jsonify({'error': 'Campo "preco" é obrigatório.'}), 400

    connection = conecta()
    cursor = connection.cursor()

    cmd_update = "UPDATE acoes_b3 SET preco = %s WHERE id = %s"

    try:
        cursor.execute(cmd_update, (novo_preco, id))
        connection.commit()

        if cursor.rowcount == 0:
            return jsonify({'message': 'ID não encontrado.'}), 404

        return jsonify({'message': f'Preço da ação ID {id} atualizado com sucesso!'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        encerra_conexao(connection)
        

#delete
@app.route('/acoes/deletar/<int:id>', methods=['DELETE'])
def deleta(id):
    connection = conecta()
    cursor = connection.cursor()

    cmd_delete = "DELETE FROM acoes_b3 WHERE id = %s"

    try:
        cursor.execute(cmd_delete, (id,))
        connection.commit()

        if cursor.rowcount == 0:
            return jsonify({'mesage': f'ID {id} não encontrado.'}), 404
        
        return jsonify({'message': f'ID {id} excluido com sucesso!'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        encerra_conexao(connection)


app.run(port=5001, host='localhost', debug=True)