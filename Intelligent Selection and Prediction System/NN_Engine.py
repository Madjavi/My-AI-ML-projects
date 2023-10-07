# Imported libraries.
import csv
import os
import random
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from os import system, name
from keras.callbacks import Callback
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

"""This library is the back end of the operation the front end of the operation is the Expert system."""


# This class config houses all the variables for the clr_t() function.
class config:
    sys1 = 'nt'
    sys2 = 'clear'
    sys3 = 'cls'


# This function clears the console terminal.
def clr_T():
    if name == config.sys1:
        _ = system(config.sys3)
    else:
        _ = system(config.sys2)


# This class generates a random list of candidates to create a csv file.
class data_GEN:
    LName = ['Chapman', 'Smith', 'Perry', 'Jones', 'Jimenez', 'Sullivan', 'Stevens', 'Turner', 'Brown', 'Ferguson',
             'Harris', 'Tanner', 'Wells', 'Grant', 'Allen', 'Harper', 'Wright', 'Johnston', 'Phillips', 'Johnson',
             'Patterson', 'Lopez', 'Lee', 'Jackson', 'Connor', 'OConnor', 'OBannon', 'Moreno', 'Negron', 'Valdez',
             'Franklin', 'DiBlacio', 'Pacheco', 'Velazquez', 'Coldero', 'Beltran', 'Higgins', 'Armstrong', 'Graig',
             'Myers']

    Genders = ['Male', 'Female']

    LTraits = ['Integrity', 'Confidence', 'Inspiration', 'Passion', 'Innovation', 'Humility', 'Discipline',
               'Self-Motivates', 'Ethical', 'Accountability', 'Courage', 'Empowerment', 'Strategic Thinking',
               'Mentoring', 'Selfless', 'Delegation', 'Communication', 'Self-Awareness', 'Gratitude',
               'Emphatic', 'Influence', 'Good Negotiator', 'Performance Driven', 'Conviction',
               'Sense of Direction', 'Adaptability', 'Risk-taking', 'Team-building', 'Emotional stability',
               'Innovative', 'Honest', 'Visionary', 'Problem-Solving Skills', 'Fair Attitude', 'Inquisitiveness',
               'Care for Others', 'Emotional Intelligence', 'Resilience']

    structure = {'Last': [], 'TagNum': [], 'Gender': [], 'Age': [], 'traits': [], 'traits_percentages': []}

    for data in LName:
        structure['Last'].append(data)
        structure['TagNum'].append(random.randint(100, 2000))
        structure['Gender'].append(random.choice(Genders))
        structure['Age'].append(random.randint(18, 30))
        structure['traits'].append(random.choice(LTraits))
        structure['traits_percentages'].append(random.uniform(0, 100))

    data_Frame_df = pd.DataFrame(structure)


# This class generates the csv dataset file that later on be used to generate a tensor dataset.
class dataset_File_GEN:
    with open('L_Dataset.csv', mode='w') as file:
        create = csv.writer(file)
        create.writerow(['Last', 'TagNum', 'Gender', 'Age', 'traits', 'traits_percentages'])

        for data2 in data_GEN.LName:
            create.writerow([random.choice(data2), random.randint(100, 2000), random.choice(data_GEN.Genders),
                             random.randint(18, 30), random.choice(data_GEN.LTraits), random.uniform(0, 100)])


# This class loads the csv dataset file to convert it into a TensorFlow dataset that will be used to train the ANN.
class dataset_pre_process:
    df_data = pd.read_csv('L_Dataset.csv')

    df_data = df_data.drop(['Last', 'Gender', 'traits'], axis=1)

    df_data = df_data.fillna(df_data.mean())

    scaler = MinMaxScaler()
    df_data[['TagNum', 'traits_percentages']] = scaler.fit_transform(df_data[['TagNum', 'traits_percentages']])

    encoder = OneHotEncoder()
    encoded_data = encoder.fit_transform(df_data[['Age']]).toarray()
    df_data = pd.concat([df_data, pd.DataFrame(encoded_data)], axis=1)
    df_data = df_data.drop(['Age'], axis=1)


# This class creates a tensor csv file.
class create_new_Tensor_dataset:
    tensor_dataset = pd.DataFrame({'TagNum': dataset_pre_process.df_data['TagNum'],
                                   'traits_percentages': dataset_pre_process.df_data['traits_percentages']})
    tensor_dataset.to_csv('tensor_dataset.csv', index=False)


# This class contains the ANN and a Callback function that stops the epoch log from showing in the console.
class ANN(Callback):
    clr_T()

    def on_epoch_end(self, epoch, logs=None):
        pass

    checkpoint_path = "L_Dataset.csv"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor="val_loss",
                                                     save_weights_only=False, verbose=0, save_freq='epoch',
                                                     save_best_only=False)

    train_data, test_data, train_labels, test_labels = train_test_split(
        dataset_pre_process.df_data.drop('TagNum', axis=1), dataset_pre_process.df_data['traits_percentages'],
        test_size=0.2)

    Leadership_model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    Leadership_model.compile(optimizer='adam', loss='mean_squared_error', metrics='accuracy')

    training = Leadership_model.fit(train_data, train_labels, epochs=100, validation_split=0.2, verbose=0)

    data_history = training.history

    test_loss = Leadership_model.evaluate(test_data, test_labels)
    predictions = Leadership_model.predict(train_data)


# This function provides a visual representation of the accuracy of the model.
def plot_training_data():
    plt.plot(ANN.data_history['accuracy'])
    plt.plot(ANN.data_history['val_accuracy'])
    plt.title('TensorFlow Model Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # value loss plot

    plt.plot(ANN.data_history['loss'], label='loss')
    plt.plot(ANN.data_history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.show()


# This class uses the prediction results from the TensorFlow model to convert them back to string then to characters
# in order to make possible the selection process that is based on leadership traits.
class candidate_Selection:
    results = ANN.predictions
    floats_array = results
    convert_to_string = [str(x) for x in floats_array]
    convert_to_char = [char for string in convert_to_string for char in string]

    for convert_to_string in range(1):
        for char in convert_to_char:

            if char == '8' or char == '5' or char == '0':
                L_Name = random.sample(data_GEN.LName, 11)

            if char == '2' or char == '7' or char == '3':
                LTraits_1 = ['Integrity', 'Confidence', 'Inspiration', 'Passion', 'Innovation', 'Humility',
                             'Discipline',
                             'Self-Motivates', 'Ethical', 'Accountability', 'Courage', 'Empowerment',
                             'Strategic Thinking',
                             'Mentoring', 'Selfless', 'Delegation', 'Communication', 'Self-Awareness', 'Gratitude',
                             'Emphatic', 'Influence', 'Good Negotiator', 'Performance Driven', 'Conviction',
                             'Sense of Direction', 'Adaptability', 'Risk-taking', 'Team-building',
                             'Emotional stability',
                             'Innovative', 'Honest', 'Visionary', 'Problem-Solving Skills', 'Fair Attitude',
                             'Inquisitiveness',
                             'Care for Others', 'Emotional Intelligence', 'Resilience']

                repeat_num = 11

                for _ in range(repeat_num):
                    L_Trait_1 = LTraits_1

            if char == '9' or char == '1' or char == '4':
                LTraits_2 = ['Integrity', 'Confidence', 'Inspiration', 'Passion', 'Innovation', 'Humility',
                             'Discipline',
                             'Self-Motivates', 'Ethical', 'Accountability', 'Courage', 'Empowerment',
                             'Strategic Thinking',
                             'Mentoring', 'Selfless', 'Delegation', 'Communication', 'Self-Awareness', 'Gratitude',
                             'Emphatic', 'Influence', 'Good Negotiator', 'Performance Driven', 'Conviction',
                             'Sense of Direction', 'Adaptability', 'Risk-taking', 'Team-building',
                             'Emotional stability',
                             'Innovative', 'Honest', 'Visionary', 'Problem-Solving Skills', 'Fair Attitude',
                             'Inquisitiveness',
                             'Care for Others', 'Emotional Intelligence', 'Resilience']

                repeat_num2 = 11

                for _ in range(repeat_num2):
                    L_Trait_2 = LTraits_2


class data_Frame:
    new_Structure = {'Last:': [], 'Tag:': [], 'Sex:': [], 'Age:': [], 'trait1:': [], 'trait2:': []}

    for data_2 in candidate_Selection.L_Name:
        new_Structure['Last:'].append(data_2)
        new_Structure['Tag:'].append(random.randint(100, 2000))
        new_Structure['Sex:'].append(random.choice(data_GEN.Genders))
        new_Structure['Age:'].append(random.randint(18, 30))
        new_Structure['trait1:'].append(random.choice(candidate_Selection.L_Trait_1))
        new_Structure['trait2:'].append(random.choice(candidate_Selection.L_Trait_2))

    Frame_data_df = pd.DataFrame(new_Structure)
