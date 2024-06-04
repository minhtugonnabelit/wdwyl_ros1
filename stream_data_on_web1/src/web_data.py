# from flask import Flask, jsonify, render_template, request, redirect, url_for, session, flash
# import rospy
# from std_msgs.msg import String
# import socket
# from threading import Thread
# import time
# from werkzeug.security import check_password_hash, generate_password_hash
# from flask_wtf import FlaskForm
# from wtforms import StringField, PasswordField
# from wtforms.validators import InputRequired
# from stream_data_on_web.srv import CurrentState, CurrentStateResponse

# app = Flask(__name__)
# app.secret_key = 'a_really_secure_secret_key'  # Change to a real secure key in production

# # Simple form definition using Flask-WTF
# class LoginForm(FlaskForm):
#     userID = StringField('UserID', validators=[InputRequired()])
#     password = PasswordField('Password', validators=[InputRequired()])

# # Hashed credentials for comparison, using werkzeug's password hashing
# USER_ID = 'admin'
# PASSWORD_HASH = generate_password_hash('password')

# @app.route('/', methods=['GET', 'POST'])
# def login():
#     if 'logged_in' in session:
#         if session['logged_in']:
#             return redirect(url_for('home'))  # Redirect if already logged in
#         session.pop('logged_in', None)  # Clean up just in case

#     form = LoginForm()
#     if form.validate_on_submit():
#         if form.userID.data == USER_ID and check_password_hash(PASSWORD_HASH, form.password.data):
#             session['logged_in'] = True
#             flash('You have successfully logged in.', 'success')
#             return redirect(url_for('home'))
#         else:
#             flash('Invalid userID or password', 'error')
#     return render_template('login.html', form=form)

# @app.route('/home')
# def home():
#     if not session.get('logged_in'):
#         return redirect(url_for('login'))
#     data = data_store.get_data()
#     return render_template('fetch.html', title='Robotics Studio 2: Bottle sorting', ros_data=data['ros_data'])

# @app.route('/data')
# def data():
#     if not session.get('logged_in'):
#         return jsonify({'error': 'Authentication required'}), 401
#     response = jsonify(data_store.get_data())
#     response.headers.add('Cache-Control', 'no-cache, no-store, must-revalidate')
#     response.headers.add('Pragma', 'no-cache')
#     response.headers.add('Expires', '0')
#     return response

# # @app.route('/state')
# # def show_state():
# #     if not session.get('logged_in'):
# #         return redirect(url_for('login'))
# #     return render_template('state.html', state=data_store.get_data()['ros_data'])

# @app.route('/logout')
# def logout():
#     session.pop('logged_in', None)
#     flash('You have been logged out.', 'info')
#     return redirect(url_for('login'))

# # def state_service():
# #     # rospy.init_node('state_service_node')
# #     s = rospy.Service('current_state', CurrentState, handle_current_state)
# #     rospy.spin()

# def handle_current_state(req):
#     data_store.update_data(req.input)
#     return CurrentStateResponse("Current state: " + req.input)

# class DataStore:
#     def __init__(self):
#         from threading import Lock
#         self._lock = Lock()
#         self._data = {'ros_data': 'No data yet'}

#     def update_data(self, message):
#         with self._lock:
#             self._data['ros_data'] = message

#     def get_data(self):
#         with self._lock:
#             return self._data.copy()

# data_store = DataStore()

# def ros_callback(message):
#     data_store.update_data(message.data)

# def get_ip_address():
#     s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#     try:
#         s.connect(('10.255.255.255', 1))
#         IP = s.getsockname()[0]
#     except Exception:
#         IP = '127.0.0.1'
#     finally:
#         s.close()
#     return IP

# def run_flask():
#     app.run(host=get_ip_address(), port=5000, debug=False, use_reloader=False, threaded=True)

# if __name__ == '__main__':
#     rospy.init_node('state_service_node', anonymous=True)
#     rospy.Subscriber('chatter', String, ros_callback)
#     rospy.Service('current_state', CurrentState, handle_current_state)

#     flask_thread = Thread(target=run_flask)
#     flask_thread.start()

#     try:
#         rospy.spin()
#     except KeyboardInterrupt:
#         pass
#     finally:
#         flask_thread.join()  # Ensure Flask thread exits cleanly

from flask import Flask, jsonify, render_template, request, redirect, url_for, session, flash
import rospy
from std_msgs.msg import String
import socket
from threading import Thread
from werkzeug.security import check_password_hash, generate_password_hash
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField
from wtforms.validators import InputRequired
from stream_data_on_web.srv import CurrentState, CurrentStateResponse

app = Flask(__name__)
app.secret_key = 'a_really_secure_secret_key'

class LoginForm(FlaskForm):
    userID = StringField('UserID', validators=[InputRequired()])
    password = PasswordField('Password', validators=[InputRequired()])

USER_ID = 'admin'
PASSWORD_HASH = generate_password_hash('password')

@app.route('/', methods=['GET', 'POST'])
def login():
    if 'logged_in' in session:
        if session['logged_in']:
            return redirect(url_for('home'))
        session.pop('logged_in', None)

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
    return render_template('fetch.html', title='Robotics Studio 2: Bottle sorting', ros_data=data)

@app.route('/data')
def data():
    if not session.get('logged_in'):
        return jsonify({'error': 'Authentication required'}), 401
    response = jsonify(data_store.get_data())
    response.headers.add('Cache-Control', 'no-cache, no-store, must-revalidate')
    response.headers.add('Pragma', 'no-cache')
    response.headers.add('Expires', '0')
    return response

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

def handle_current_state(req):
    data_store.update_data(req.input)
    return CurrentStateResponse("Current state: " + req.input)

class DataStore:
    def __init__(self):
        from threading import Lock
        self._lock = Lock()
        self._data = {
            'num_bottles': 0,
            'classification': '',
            'picked_bottle': '',
            'accuracy': 0.0,
            'bottle_left' : 0
        }
# rostopic section
    def update_data(self, topic, message):
        with self._lock:
            if topic == '/num_bottles':
                self._data['num_bottles'] = int(message.data)
            elif topic == '/bottle_classification':
                self._data['classification'] = message.data
            elif topic == '/picked_bottle':
                self._data['picked_bottle'] = message.data
            elif topic == '/picking_accuracy':
                self._data['accuracy'] = float(message.data)
            elif topic == '/bottle_left':
                self._data['bottle_left'] = int(message.data)

    def get_data(self):
        with self._lock:
            return self._data.copy()

data_store = DataStore()

def ros_callback_num_bottles(message):
    data_store.update_data('/num_bottles', message)

def ros_callback_classification(message):
    data_store.update_data('/bottle_classification', message)

def ros_callback_picked_bottle(message):
    data_store.update_data('/picked_bottle', message)

def ros_callback_picking_accuracy(message):
    data_store.update_data('/picking_accuracy', message)

def ros_callback_bottle_left(message):
    data_store.update_data('/bottle_left', message)

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
    app.run(host=get_ip_address(), port=5000, debug=False, use_reloader=False, threaded=True)

if __name__ == '__main__':
    rospy.init_node('state_service_node', anonymous=True)
    rospy.Subscriber('/num_bottles',String, ros_callback_num_bottles)
    rospy.Subscriber('/bottle_classification', String, ros_callback_classification)
    rospy.Subscriber('/picked_bottle', String, ros_callback_picked_bottle)
    rospy.Subscriber('/picking_accuracy', String, ros_callback_picking_accuracy)
    rospy.Subscriber('/bottle_left', String, ros_callback_bottle_left)
    rospy.Service('current_state', CurrentState, handle_current_state)

    flask_thread = Thread(target=run_flask)
    flask_thread.start()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass
    finally:
        flask_thread.join()
