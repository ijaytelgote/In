from flask import Flask, jsonify

# Create a Flask application instance
app = Flask(__name__)

# Define a route to the home page
@app.route('/')
def home():
    return "Welcome to the Flask App!"

# Define a route that returns a JSON response
@app.route('/api/data')
def get_data():
    data = {
        'message': 'Hello, this is a Flask app',
        'success': True
    }
    return jsonify(data)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
