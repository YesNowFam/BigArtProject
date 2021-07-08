from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import cv2
import numpy as np

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
from keras.models import Model

model = VGG16(weights='imagenet', include_top=True)
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

def compute_features(path):
    image = load_img(path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    return model.predict(image)[0]

#y = []
#X = []

#with open('scores.txt', 'r') as s:
    #scores = s.readlines()

#for i in range(188):
    #print(i)
    #f = compute_features(f'images/{i}.bmp')
    #X.append(f)
    #print(h)
    #s = int(scores[i].rstrip())
    #y.append([s])
    #print(s)

#np.save('X.npy', X)
#np.save('y.npy', y)
X = np.load('X.npy')
y = np.load('y.npy')

sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

gsc = GridSearchCV(
        estimator=SVR(kernel='rbf'),
        param_grid={
            'C': np.logspace(-2, 10, 13),
            'epsilon': np.logspace(-11, 1, 13),
            'gamma': np.logspace(-9, 3, 13)
        },
        cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)
grid_result = gsc.fit(x_train, np.ravel(y_train))
best_params = grid_result.best_params_
print(best_params)

regressor = SVR(kernel='rbf', C=best_params["C"], epsilon=best_params["epsilon"], gamma=best_params["gamma"],
                   coef0=0.1, shrinking=True,
                   tol=0.001, cache_size=200, verbose=False, max_iter=-1)
regressor.fit(x_train, np.ravel(y_train))

y_pred = regressor.predict(x_test)
y_pred = sc_y.inverse_transform(y_pred)
print(y_pred)
print(np.sqrt(mean_squared_error(y_test, y_pred)))

#joblib.dump(regressor, 'model.pkl')
#regressor = joblib.load('model.pkl')
