from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from flask_sqlalchemy import SQLAlchemy
import random
import string
import os, json

# Explicitly set the template folder
# template_dir = os.path.abspath('./templates')

app = Flask(__name__) # , template_folder=template_dir)
app.config['SECRET_KEY'] = 'secret!'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///charts.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
socketio = SocketIO(app)

# Define the Chart model
class Chart(db.Model):
    id = db.Column(db.String(8), primary_key=True)
    training_losses = db.Column(db.Text, nullable=True)
    eval_losses = db.Column(db.Text, nullable=True)

    def __init__(self, id):
        self.id = id
        self.training_losses = "[]"
        self.eval_losses = "[]"

    def update_losses(self, training_losses, eval_losses):
        self.training_losses = str(training_losses)
        self.eval_losses = str(eval_losses)

# Initialize the database
with app.app_context():
    db.create_all()

@app.route('/')
def index():
    return "Welcome to the Training Loss Dashboard!"

@app.route('/chart/<chart_id>')
def chart(chart_id):
    chart = Chart.query.get(chart_id)
    if chart:
        training_losses = json.loads(chart.training_losses)
        eval_losses = json.loads(chart.eval_losses)
        return render_template('chart.html', chart_id=chart_id, training_losses=training_losses, eval_losses=eval_losses)
    else:
        return "Chart not found", 404

@app.route('/create_chart', methods=['POST'])
def create_chart():
    name = request.json.get('name', None)
    if not name:
        name = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
    
    existing_chart = Chart.query.filter_by(id=name).first()
    if existing_chart:
        return jsonify({"chart_id": name})

    new_chart = Chart(id=name)
    db.session.add(new_chart)
    db.session.commit()
    
    return jsonify({"chart_id": name})

@app.route('/update_chart/<chart_id>', methods=['POST'])
def update_chart(chart_id):
    chart = Chart.query.get(chart_id)
    if not chart:
        return jsonify({"error": "Chart ID not found"}), 404
    
    data = request.json
    training_losses = eval(chart.training_losses)
    eval_losses = eval(chart.eval_losses)
    
    if 'loss' in data:
        training_losses.append(data['loss'])
    if 'eval_loss' in data:
        eval_losses.append(data['eval_loss'])
    
    chart.update_losses(training_losses, eval_losses)
    db.session.commit()
    
    # Emit update to client
    socketio.emit('update_chart', { "chart_id": chart_id, "data": {"training_losses": training_losses, "eval_losses": eval_losses} })
    return jsonify({"status": "success"})

if __name__ == '__main__':
    socketio.run(app, debug=True)
