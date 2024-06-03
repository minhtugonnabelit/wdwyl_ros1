from flask import Flask, jsonify, render_template, request, redirect, url_for, session, flash
import rospy
from std_msgs.msg import String
import socket
from threading import Thread
import time
from werkzeug.security import check_password_hash, generate_password_hash
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField
from wtforms.validators import InputRequired
from stream_data_on_web.srv import CurrentState, CurrentStateResponse
from stream_data_on_web.srv import NumberOfBottle, NumberOfBottleResponse

app = Flask(__name__)
app.secret_key = 'a_really_secure_secret_key'  # Change to a real secure key in production

# Simple form definition using Flask-WTF
class LoginForm(FlaskForm):
    userID = StringField('UserID', validators=[InputRequired()])
    password = PasswordField('Password', validators=[InputRequired()])

# Hashed credentials for comparison, using werkzeug's password hashing
USER_ID = 'admin'
PASSWORD_HASH = generate_password_hash('password')

@app.route('/', methods=['GET', 'POST'])
def login():
    if 'logged_in' in session:
        if session['logged_in']:
            return redirect(url_for('home'))  # Redirect if already logged in
        session.pop('logged_in', None)  # Clean up just in case

    form = LoginForm()
    if form.validate_on_submit():
        if form.userID.data == USER_ID and check_password_hash(PASSWORD_HASH, form.password.data):
            session['logged_in'] = True
            flash('You have successfully logged in.', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid userID or password', 'error')
    return render_template('login.html', form=form)

@app.route('/home')
def home():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    data = data_store.get_data()
    return render_template('fetch.html', title='Robotics Studio 2: Bottle sorting', 
                           warning_data=data['warning_data'], state_data=data['state_data'], number_of_bottle=data['number_of_bottle'], type_of_bottle=['type_of_bottle'],
                           number_of_crown_bottle=data['number_of_crown_bottle'], number_of_heineken_bottle=data['number_of_heineken_bottle'],
                           number_of_great_northern_bottle=data['number_of_great_northern_bottle'], number_of_4_pines_bottle=data['number_of_4_pines_bottle'])

@app.route('/data')
def data():
    if not session.get('logged_in'):
        return jsonify({'error': 'Authentication required'}), 401
    response = jsonify(data_store.get_data())
    # response.headers.add('Cache-Control', 'no-cache, no-store, must-revalidate')
    # response.headers.add('Pragma', 'no-cache')
    # response.headers.add('Expires', '0')
    return response

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

# def state_service():
#     # rospy.init_node('state_service_node')
#     s = rospy.Service('current_state', CurrentState, handle_current_state)
#     rospy.spin()
def warning_data (req):
    data_store.update_data("WARNING: " + req.input, data_type='warning_data')
    return CurrentStateResponse("WARNING: " + req.input)

def handle_current_state(req):
    data_store.update_data("Current state: " + req.input, data_type='state_data')
    return CurrentStateResponse("Current state: " + req.input)

def handle_number_of_bottle_in_crate(req):
    data_store.update_data("Number of bottle: " + req.input, data_type='number_of_bottle')
    return CurrentStateResponse("Number of bottle: " + req.input)

def type_of_bottle(req):
    data_store.update_data("Type of bottle: " + req.input, data_type='type_of_bottle')
    return CurrentStateResponse("Type of bottle: " + req.input)

def number_of_crown_bottle (req):
    data_store.update_data("Number of sorted crown bottle: " + req.input, data_type='number_of_crown_bottle')
    return CurrentStateResponse("Number of sorted crown bottle: " + req.input)

def number_of_heineken_bottle (req):
    data_store.update_data("Number of sorted heineken bottle: " + req.input, data_type='number_of_heineken_bottle')
    return CurrentStateResponse("Number of sorted heineken bottle: " + req.input)

def number_of_4_pines_bottle (req):
    data_store.update_data("Number of sorted 4 pines bottle: " + req.input, data_type='number_of_4_pines_bottle')
    return CurrentStateResponse("Number of sorted 4 pines bottle: " + req.input)

def number_of_great_northern_bottle (req):
    data_store.update_data("Number of sorted great northern bottle: " + req.input, data_type='number_of_great_northern_bottle')
    return CurrentStateResponse("Number of sorted great norther bottle: " + req.input)

class DataStore:
    def __init__(self):
        from threading import Lock
        self._lock = Lock()
        self._data = {'warning_data': 'NO WARNING',
                      'state_data': 'Current state:',
                      'number_of_bottle': 'Number of bottle:',
                      'type_of_bottle': 'Type of bottle',
                      'number_of_crown_bottle': 'Number of sorted crown bottle: 0',
                      'number_of_heineken_bottle': 'Number of sorted heineken bottle: 0',
                      'number_of_4_pines_bottle': 'Number of sorted 4 pines bottle: 0',
                      'number_of_great_northern_bottle': 'Number of sorted great northern bottle: 0'}

    def update_data(self, message, data_type='warning_data'):
        with self._lock:
            self._data[data_type] = message

    def get_data(self):
        with self._lock:
            return self._data.copy()

data_store = DataStore()

def ros_callback(message):
    data_store.update_data(message.data)

def get_ip_address():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

def run_flask():
    # app.run(host=get_ip_address(), port=5000, debug=False, use_reloader=False, threaded=True)
    app.run(host='192.168.1.127', port=5000, debug=False, use_reloader=False, threaded=True)

if __name__ == '__main__':
    rospy.init_node('state_service_node', anonymous=True)
    rospy.Service('warning_data', CurrentState, warning_data)
    rospy.Service('current_state', CurrentState, handle_current_state)
    rospy.Service('number_of_bottle', CurrentState, handle_number_of_bottle_in_crate)
    rospy.Service('type_of_bottle', CurrentState, type_of_bottle)
    rospy.Service('number_of_crown_bottle', CurrentState, number_of_crown_bottle)
    rospy.Service('number_of_heineken_bottle', CurrentState, number_of_heineken_bottle)
    rospy.Service('number_of_4_pines_bottle', CurrentState, number_of_4_pines_bottle)
    rospy.Service('number_of_great_northern_bottle', CurrentState, number_of_great_northern_bottle)
    # rospy.Subscriber('chatter', String, ros_callback)

    flask_thread = Thread(target=run_flask)
    flask_thread.start()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass
    finally:
        flask_thread.join()  # Ensure Flask thread exits cleanly
        # pass

