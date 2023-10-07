# Imported libraries.
import time as tm
import NN_Engine as NN  # This library is the engine that houses all the backend functionality.
from pyknow import *

"""This expert system is the front end of the operation the back end of the operation is the NN_Engine library.
   This expert system has been debug in order to make it work with current versions of python. """


# All the classes below houses the "fact" variables for analysis.
class Interaction(Fact):
    """For the different options required to run the Algorithm"""
    pass


class TensorFlow(Fact):
    """To Ask and interact with the predictions"""


class Predictions(Fact):
    """Interacts with the prediction of each of the candidates based on their leadership traits"""


class Not_Interested(Fact):
    """Just for quit and exit the program execution"""


# This class houses all the labels and informational data of the expert system.
class the_Info_Pool:
    ver = NN.tf.__version__

    Info1 = {'Information': "\nWelcome to the AI Leadership Selection and Prediction System\n",

             'Information2': "\nThe Algorithm generates a dataset from a pool of candidate that then it is ingested"
                             "\ninto an AI TensorFlow Feedforward neural network model that trains and analyzes "
                             "\nthe data to generates a prediction that serves to select the most qualified "
                             "\ncandidates based on leadership traits.\n"
                             "\nThe Instructions goes as follows:",

             'Information3': "\nWhat are Leadership Traits?\n" 
                             "\nLeadership traits refer to personal qualities that define effective leaders. "
                             "\nLeadership refers to the ability of an individual or an organization to "
                             "\nguide individuals, teams, or organizations toward the fulfillment of "
                             "\ngoals and objectives. It plays an important function in management, "
                             "\nas it helps maximize efficiency and achieve strategic and organizational goals. "
                             "\nLeaders help motivate others, provide guidance, build morale, "
                             "\nimprove the work environment, and initiate action (Wale H., 2019).",

             'Instructions': "\n1. Information & Instructions - Briefly describe how the model works."
                             "\n2. View Candidates Data - Provide a list of the candidates' data before conversion."
                             "\n3. Generate Candidates Dataset - Creates a CSV dataset to convert into a tensor."
                             "\n4. Generate a Tensor Dataset - Creates a tensor dataset from the CSV dataset."
                             "\n5. Train the TensorFlow Model - Pass the tensor dataset to train the model."
                             "\n6. Generate and view plot graph - Generates an accuracy/loss plot graph."
                             "\n7. Generate Prediction Results - Generates the candidates prediction results."
                             "\n8. Quit - Exit the model.\n"
                             "\nAll the interaction is accomplished by answering yes or no "
                             "and selecting one of the options ranging from 1 to 8.\n"
                             "\nNOTE: The model is pre-trained, you will notice at the beginning of the execution, but "
                             "\nby selecting option number 5, the model gets re-trained; "
                             "\nthe model can be trained many times over.\n"
                             "\nHave Fun and Enjoy!\n",

             'Tag': f"Powered by: TensorFlow Version: {ver} & Expert Systems Interactions 2.0"}

    Interaction1 = {'Data1': "\nThere's Nothing Really Left To Explore.\n",
                    'Data2': "\nDo you want to proceed? y/n: ",
                    'Data3': "\nHi there! Would you like to begin? y/n: ",
                    'Data4': "\nIn Case I Don't See Ya, Good Afternoon, Good Evening And Goodnight.\n"}

    selection = {'Option1': "\n1. Information & Instructions",
                 'Option2': "2. View Candidates Data",
                 'Option3': "3. Generate Candidates Dataset",
                 'Option4': "4. Generate a Tensor Dataset",
                 'Option5': "5. Train the TensorFlow Model",
                 'Option6': "6. Generate and view plot graph",
                 'Option7': "7. Generate Prediction Results",
                 'Option8': "8. Quit",
                 'Prompt': "\nEnter Option: "}


# This class is the knowledge engine that contains all the rules of the expert system required for the back-end
# function.
class Main_ANN_Core(KnowledgeEngine):
    NN.clr_T()

    @Rule(Interaction(usrInteractio='y'))
    def Informational(self):
        print(the_Info_Pool.Info1['Information'])
        print(the_Info_Pool.Info1['Tag'])
        print(the_Info_Pool.selection['Option1'])
        print(the_Info_Pool.selection['Option2'])
        print(the_Info_Pool.selection['Option3'])
        print(the_Info_Pool.selection['Option4'])
        print(the_Info_Pool.selection['Option5'])
        print(the_Info_Pool.selection['Option6'])
        print(the_Info_Pool.selection['Option7'])
        print(the_Info_Pool.selection['Option8'])

    @Rule(Interaction(selection='1'))
    def Instructions(self):
        print(the_Info_Pool.Info1['Information3'])
        print(the_Info_Pool.Info1['Information2'])
        print(the_Info_Pool.Info1['Instructions'])

    @Rule(TensorFlow(selection2='2'))
    def DataView(self):
        tm.sleep(2)
        dataGen = NN.data_GEN.data_Frame_df
        dataView = f'\nInput Data Listed Below: \n\n{dataGen}'
        print(dataView)
        tm.sleep(2)

    @Rule(TensorFlow(selection3='3'))
    def DatasetGen(self):
        tm.sleep(2)
        NN.dataset_File_GEN()
        dataSet_MSG1 = 'Dataset file ready!'
        dataSet_MSG2 = f'\nDataset file has been generated....\n\n{dataSet_MSG1}'
        print(dataSet_MSG2)

    @Rule(TensorFlow(selection4='4'))
    def DataLoading(self):
        tm.sleep(2)
        NN.dataset_pre_process()
        NN.create_new_Tensor_dataset()
        tensor_dataset = NN.dataset_pre_process.df_data
        tensor_preview = f'\nTensor dataset generated.......\n\n{tensor_dataset}'
        print(tensor_preview)

    @Rule(TensorFlow(selection5='5'))
    def AI_Training(self):
        tm.sleep(2)
        model_summary = NN.ANN.Leadership_model.summary()
        tm.sleep(2)
        model_History = NN.ANN.test_loss
        tm.sleep(2)
        model_Prediction = NN.ANN.predictions
        tm.sleep(2)
        model_callbacks = NN.ANN.cp_callback
        tm.sleep(2)
        Neural_Process1 = f'\nInitiating Artificial Neural Network Training........\n\n{model_summary}'
        tm.sleep(2)
        Neural_Process2 = f'\nNeural Network Process History: {model_History}\n'
        tm.sleep(2)
        Neural_Prediction = f'\nModel Predictions Results: \n\n{model_Prediction}\n'
        callbacks_obj = f'\nCallback object shows as follows: \n{model_callbacks}\n'
        print(Neural_Process1, Neural_Process2, Neural_Prediction, callbacks_obj)
        tm.sleep(2)

    @Rule(Predictions(selection6='6'))
    def Model_Prediction(self):
        tm.sleep(2)
        plot_graph = NN.plot_training_data()
        plot_graph_MSG = "\nGenerating Plot Graph.......\n\nComplete!"
        print(plot_graph_MSG, plot_graph)
        tm.sleep(2)

    @Rule(Predictions(selection7='7'))
    def Candidate_Selection(self):
        tm.sleep(2)
        selected_Candidate = NN.data_Frame.Frame_data_df
        Selection_Process = '\nThe following candidate has been selected as strategic leaders:\n\n'
        print(Selection_Process, selected_Candidate)
        tm.sleep(2)

    @Rule(Not_Interested(selection8='8'))
    def I_Just_want_to_quit(self):
        print(the_Info_Pool.Interaction1['Data1'])
        quit()
        tm.sleep(2)

    @Rule(Interaction(usrInteraction2='n'))
    def Noting_else_to_do(self):
        print(the_Info_Pool.Interaction1['Data4'])
        quit()
        tm.sleep(2)


# This class houses all the engine related executions to run the knowledge engine.
class deploy:
    NN.clr_T()

    engine = Main_ANN_Core()
    engine.reset()

    engine1 = Main_ANN_Core()
    engine1.reset()

    engine2 = Main_ANN_Core()
    engine2.reset()

    engine3 = Main_ANN_Core()
    engine3.reset()

    engine4 = Main_ANN_Core()
    engine4.reset()

    ask_Information = input(the_Info_Pool.Interaction1['Data3'])

    while True:
        ask_Information = input(the_Info_Pool.Interaction1['Data2'])

        engine.declare(Interaction(usrInteractio=ask_Information), (Interaction(usrInteraction2=ask_Information)))
        engine.run()

        usr_Int = input(the_Info_Pool.selection['Prompt'])

        engine1.declare(Interaction(selection=usr_Int), (TensorFlow(selection2=usr_Int)))
        engine1.run()

        engine2.declare(TensorFlow(selection3=usr_Int), (TensorFlow(selection4=usr_Int)))
        engine2.run()

        engine3.declare(TensorFlow(selection5=usr_Int), (Predictions(selection6=usr_Int)))
        engine3.run()

        engine4.declare(Predictions(selection7=usr_Int), (Not_Interested(selection8=usr_Int)))
        engine4.run()
