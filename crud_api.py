from database import conecta, encerra_conexao
from flask import Flask, jsonify, request

app = Flask(__name__)

#consulta
@app.route('/cadastroFuncionario/consulta', methods=['GET'])
def consultar():
    connection = conecta()
    cursor = connection.cursor()

    cmd_select = "SELECT id , name, setor, gender, work_life_balance, marital_status, job_level, remote_work, attrition, created_at FROM cadastroFuncionario;"

    try:
        cursor.execute(cmd_select)
        resultado = cursor.fetchall()

        dados = []

        for dado in resultado:
            dados.append({
                'id': dado[0],
	            'name': dado[1],
                'setor': dado[2],
                'gender': dado[3],
                'work_life_balance': dado[4],
                'marital_status': dado[5],
                'job_level': dado[6],
                'remote_work': dado[7],
                'attrition': dado[8],
                'created_at': dado[9].isoformat() if dado[9] else None
            })
        return jsonify(resultado)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        encerra_conexao(connection)


#inserir
@app.route('/cadastroFuncionario/cadastrar', methods=['POST'])
def cadastrar():
    data = request.get_json()

    name = data.get('name')
    setor = data.get('setor')
    gender = data.get('gender')
    work_life_balance = data.get('work_life_balance')
    marital_status = data.get('marital_status')
    job_level = data.get('job_level')
    remote_work = data.get('remote_work')

    connection = conecta()
    cursor = connection.cursor()

    cmd_insert = "INSERT INTO cadastroFuncionario (name, setor, gender, work_life_balance, marital_status, job_level, remote_work) VALUES (%s,%s,%s,%s,%s,%s,%s);"
    values = (name, setor, gender, work_life_balance, marital_status, job_level, remote_work)

    try:
        cursor.execute(cmd_insert, values)
        connection.commit()
        print('Dados inseridos com sucesso!')
        return jsonify({'message': 'Dados inseridos com sucesso!'}), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        encerra_conexao(connection)

app.run(port=5000, host='localhost', debug=True)