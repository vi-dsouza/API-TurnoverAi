from database import conecta, encerra_conexao
from rede_neural import buscar_dados_reais, preprocessar_dados
from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

#consulta
@app.route('/cadastroFuncionarios/consulta', methods=['GET'])
def consultar():
    connection = conecta()
    cursor = connection.cursor()

    cmd_select = "SELECT id , nome, setor, gender, work_life_balance, marital_status, job_level, remote_work, prob_permanencia, attrition, created_at FROM cadastroFuncionarios;"

    try:
        cursor.execute(cmd_select)
        resultado = cursor.fetchall()

        dados = []

        for dado in resultado:
            dados.append({
                'id': dado[0],
	            'nome': dado[1],
                'setor': dado[2],
                'gender': dado[3],
                'work_life_balance': dado[4],
                'marital_status': dado[5],
                'job_level': dado[6],
                'remote_work': dado[7],
                'prob_permanencia': dado[8],
                'attrition': dado[9],
                'created_at': dado[10].isoformat() if dado[10] else None
            })
        return jsonify(resultado)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        encerra_conexao(connection)


#inserir (esta cadastrando 1 por vez)
@app.route('/cadastroFuncionarios/cadastrar', methods=['POST'])
def cadastrar():
    data = request.get_json()

    nome = data.get('nome')
    setor = data.get('setor')
    gender = data.get('gender')
    work_life_balance = data.get('work_life_balance')
    marital_status = data.get('marital_status')
    job_level = data.get('job_level')
    remote_work = data.get('remote_work')
    prob_permanencia = data.get('prob_permanencia')
    attrition = data.get('attrition')

    connection = conecta()
    cursor = connection.cursor()

    cmd_insert = "INSERT INTO cadastroFuncionarios (nome, setor, gender, work_life_balance, marital_status, job_level, remote_work, prob_permanencia, attrition) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s);"
    values = (nome, setor, gender, work_life_balance, marital_status, job_level, remote_work, prob_permanencia, attrition)

    try:
        cursor.execute(cmd_insert, values)
        connection.commit()
        print('Dados inseridos com sucesso!')
        return jsonify({'message': 'Dados inseridos com sucesso!'}), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        encerra_conexao(connection)

#atualizar
@app.route('/cadastroFuncionarios/update/<int:id>', methods=['PUT'])
def atualizar(id):
    data = request.get_json()
    campos = ['nome', 'setor', 'gender', 'work_life_balance', 'marital_status', 'job_level', 'remote_work', 'prob_permanencia', 'attrition']
    
    campos_para_atualizar = {}
    for campo in campos:
        if campo in data:
            campos_para_atualizar[campo] = data[campo]

    if not campos_para_atualizar:
        return jsonify({'error': 'Nenhum campo válido para atualização foi enviado.'}), 400
    
    cmd_update = ", ".join([f"{campo} = %s" for campo in campos_para_atualizar])

    valores = list(campos_para_atualizar.values())
    valores.append(id)

    sql = f"UPDATE cadastroFuncionarios SET {cmd_update} WHERE id = %s"

    connection = conecta()
    cursor = connection.cursor()

    try:
        cursor.execute(sql, valores)
        connection.commit()

        if cursor.rowcount == 0:
            return jsonify({'message': 'ID não encontrado.'}), 404
        
        return jsonify({'message': f'Dados do funcionário ID {id} atualizados com sucesso!'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        encerra_conexao(connection)

#deletar
@app.route('/cadastroFuncionarios/deletar/<int:id>', methods=['DELETE'])
def excluir(id):
    connection = conecta()
    cursor = connection.cursor()

    cmd_delete = "DELETE FROM cadastroFuncionarios WHERE id = %s"

    try:
        cursor.execute(cmd_delete, (id,))
        connection.commit()

        if cursor.rowcount == 0:
            return jsonify({'message': f'ID {id} não encontrado.'}), 404
        
        return jsonify({'message': f'ID {id} excluído com sucesso!'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        encerra_conexao(connection)

@app.route('/previsao/<int:id>', methods=['GET', 'PUT'])
def previsao(id):
    try:
        # Conecta ao banco
        connection = conecta()
        cursor = connection.cursor()

        # Atualiza os dados
        buscar_dados_reais("http://localhost:5000/cadastroFuncionarios/consulta", nome_arquivo="dados_reais.xlsx")

        # Lê os dados
        df = pd.read_excel("dados_reais.xlsx")

        # Filtra pelo ID
        funcionario = df[df['Id'] == id]

        if funcionario.empty:
            return jsonify({'erro': f'Nenhum funcionário encontrado com o ID {id}'}), 404

        # Prepara os dados
        colunas_entrada = ['Gender', 'Work-Life Balance', 'Marital Status', 'Job Level', 'Remote Work']
        dados_entrada = funcionario[colunas_entrada]

        # Normalização
        scaler = joblib.load("scaler_attrition.save")
        dados_normalizados = scaler.transform(dados_entrada)

        # Carrega o modelo
        modelo = load_model('modelo_attrition.keras')

        # Faz a predição
        predicao = modelo.predict(dados_normalizados)

        # Interpreta resultado e atualiza banco
        if predicao.shape[1] == 1:
            saida = float(predicao[0][0] * 100)
            saida_formatada = f"{saida:.2f}%"  # Se quiser guardar como texto com % no banco, ou só o float, veja abaixo

            # Atualiza no banco - aqui considerando que Attrition é numérico (float) e guarda a probabilidade
            sql_update = "UPDATE cadastroFuncionarios SET prob_permanencia = %s WHERE Id = %s"
            cursor.execute(sql_update, (saida, id))
            connection.commit()

            return jsonify({'probab_permanencia': saida_formatada})

        else:
            classe = int(predicao.argmax(axis=1)[0])

            # Se quiser atualizar o banco com a classe predita:
            sql_update = "UPDATE cadastroFuncionarios SET prob_permanencia = %s WHERE Id = %s"
            cursor.execute(sql_update, (classe, id))
            connection.commit()

            return jsonify({'classe_predita': classe})

    except Exception as e:
        return jsonify({'erro': str(e)}), 500

@app.route('/permanenciaGeral', methods=['GET'])
def calculaPermanenciaGeral():
    try:
        buscar_dados_reais("http://localhost:5000/cadastroFuncionarios/consulta", nome_arquivo="dados_reais.xlsx")
        df = pd.read_excel("dados_reais.xlsx")

        permanencia_media = df["Prob_Permanencia"].mean()

        return jsonify({"permanencia_geral": f"{permanencia_media:.2f}%"})

    except Exception as e:
        return jsonify({"erro": str(e)}), 500
    
@app.route('/generoComMaiorPermanencia', methods=['GET'])
def generoComMaiorPemanencia():
    try:
        buscar_dados_reais("http://localhost:5000/cadastroFuncionarios/consulta", nome_arquivo="dados_reais.xlsx")
        df = pd.read_excel("dados_reais.xlsx")

        media_prob = df.groupby('Gender')['Prob_Permanencia'].mean()

        gener_mais_permanente = media_prob.idxmax()
        probabilidade_genero = media_prob.max()

        if gener_mais_permanente == 0:
            gener_mais_permanente = 'Feminino'
        if gener_mais_permanente == 1:
            gener_mais_permanente = 'Masculino'

        resultado_genero = f"{gener_mais_permanente} - {probabilidade_genero:.2f}%"

        return jsonify({"genero_mais_permanencte": resultado_genero})
    except Exception as e:
        return jsonify({"erro": str(e)}), 500
    
@app.route('/permanenciaPorSetor', methods=['GET'])
def setorComMaiorPermanencia():
    try:
        buscar_dados_reais("http://localhost:5000/cadastroFuncionarios/consulta", nome_arquivo="dados_reais.xlsx")
        df = pd.read_excel("dados_reais.xlsx")

        media_setor = df.groupby('Setor')['Prob_Permanencia'].mean()

        setorMaisPermanente = media_setor.idxmax()
        probabilidadeSetor = media_setor.max()

        resultadoSetor = f"{setorMaisPermanente} - {probabilidadeSetor:.2f}%"

        return jsonify({"setor_mais_permanente": resultadoSetor})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/mediaGeralSetores', methods=['GET'])
def permanenciaSetores():
    try:
        buscar_dados_reais("http://localhost:5000/cadastroFuncionarios/consulta", nome_arquivo="dados_reais.xlsx")
        df = pd.read_excel("dados_reais.xlsx")

        setores = df.groupby('Setor')['Prob_Permanencia'].mean()
        setores_formatado = setores.round(2).to_dict()

        return jsonify(setores_formatado)
    except Exception as e:
        return jsonify({"error": str(e)}), 500




app.run(port=5000, host='localhost', debug=True)