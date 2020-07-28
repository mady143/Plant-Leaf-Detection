from flask import Flask,render_template,request,redirect,url_for
from keras.preprocessing import image
# from PIL import image
import numpy as np
from deeplearning import graph, model, output_list
import base64
app = Flask(__name__)
@app.route('/')
def home():
	return render_template('index.html')
@app.route('/predict',methods = ['POST'])
def result():
	if request.method == "POST":
		myfile = request.files['myfile']
		# print("MyFileeeeeee:",myfile)
		b64_img = base64.b64encode(myfile.read()).decode('ascii')
		# print("b64_img:",b64_img)
		img = image.load_img(myfile, target_size=(224, 224))
		# print("IMG1:",img)
		img = image.img_to_array(img)
		# print("IMG2:",img)
		img = np.expand_dims(img, axis=0)
		# print("IMG3:",img)
		img = img/255
		# print("IMG4:",img)
		with graph.as_default():
			prediction = model.predict(img)
			print("prediction:",prediction)
		prediction_flatten = prediction.flatten()
		print("prediction_flatten:",prediction_flatten)
		max_val_index = np.argmax(prediction_flatten)
		print("max_val_index:",max_val_index)
		result = output_list[max_val_index]
		print("result:",result)

		return render_template('index.html',result= result, image_url= b64_img)
	return render_template('index.html')
if __name__ == '__main__':
    app.run(host = '192.168.55.103',port=5000,debug= True)
    # app.run()
    