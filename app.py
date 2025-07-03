from flask import Flask, render_template, jsonify, request
import requests
import numpy as np

app = Flask(__name__)

API_KEY = open('api_key.txt').read().strip()
ENDPOINT = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/materi')
def materi():
    return render_template('materi.html')

@app.route('/perhitungan')
def perhitungan():
    return render_template('perhitungan.html')

@app.route('/simulasi')
def simulasi():
    return render_template('simulasi.html')

@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.json.get('message')
    
    payload = {
        "contents": [{
            "parts": [{"text": f"Jawab pertanyaan tentang medan magnet: {user_input}"}]
        }]
    }

    response = requests.post(ENDPOINT, json=payload, headers={"Content-Type": "application/json"})
    if response.status_code == 200:
        data = response.json()
        try:
            reply = data['candidates'][0]['content']['parts'][0]['text']
            return jsonify({'reply': reply})
        except:
            return jsonify({'reply': 'Maaf, terjadi kesalahan parsing respon.'})
    else:
        return jsonify({'reply': 'Gagal menghubungi AI.'})

# fungsi untuk simulasi
@app.route('/calculate', methods=['POST'])
def calculate():
    try:
        data = request.json
        
        x1 = float(data.get('x1', 0))
        x2 = float(data.get('x2', 0))
        y1 = float(data.get('y1', 0))
        y2 = float(data.get('y2', 0))
        I = float(data.get('I', 0))
        n_segments = int(data.get('n_segments', 1))

        if n_segments <= 0:
            return jsonify({'error': 'Jumlah segmen harus lebih dari 0'}), 400

        if abs(x2 - x1) < 1e-6 and abs(y2 - y1) < 1e-6:
            return jsonify({'error': 'Panjang kawat tidak boleh nol'}), 400

        mu0 = 4 * np.pi * 1e-7
        
        wire_x = np.linspace(x1, x2, n_segments)
        wire_y = np.linspace(y1, y2, n_segments)
        dl_x = (x2 - x1) / n_segments
        dl_y = (y2 - y1) / n_segments
     
        X, Y = np.meshgrid(np.linspace(-2, 2, 20), np.linspace(-2, 2, 20))
        Bx = np.zeros_like(X)
        By = np.zeros_like(Y)

        for i in range(n_segments):
            rx = X - wire_x[i]
            ry = Y - wire_y[i]
            r_squared = rx**2 + ry**2
            
            safe_r_squared = np.where(r_squared < 1e-10, 1e-10, r_squared)
            r_cubed = safe_r_squared * np.sqrt(safe_r_squared)
            
            dBx = mu0 * I / (4 * np.pi) * (dl_y * rx - dl_x * ry) / r_cubed
            dBy = mu0 * I / (4 * np.pi) * (dl_x * rx + dl_y * ry) / r_cubed  
            
            Bx += dBx
            By += dBy

        return jsonify({
            "X": X.tolist(),
            "Y": Y.tolist(),
            "Bx": Bx.tolist(),
            "By": By.tolist(),
            "wire_x": wire_x.tolist(),
            "wire_y": wire_y.tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# fungsi untuk perhitungan
@app.route('/compute_point_2d', methods=['POST'])
def compute_point_2d():
    try:
        data = request.json
        I = float(data['I'])
        dlx = float(data['dlx'])
        dly = float(data['dly'])
        x_kawat = float(data['x_kawat'])
        y_kawat = float(data['y_kawat'])
        x_obs = float(data['x_obs'])
        y_obs = float(data['y_obs'])
        
        # Konstanta mu0
        mu0 = 4 * np.pi * 1e-7
        
        # Vektor r dari kawat ke titik observasi
        rx = x_obs - x_kawat
        ry = y_obs - y_kawat
        r_squared = rx**2 + ry**2
        
        if r_squared < 1e-12:
            return jsonify({'error': 'Jarak ke titik observasi terlalu dekat ke kawat'}), 400
        
        r_cubed = r_squared * np.sqrt(r_squared)
        
        # Hasil vektor medan magnet 2D (karena dlz=0, maka medan hanya keluar di sumbu z)
        # Bz keluar dari bidang x-y
        dBz = mu0 * I * (dlx * ry - dly * rx) / (4 * np.pi * r_cubed)
        
        return jsonify({
            'Bx': 0.0,
            'By': 0.0,
            'Bz': dBz
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=True)