# Stock Price Predictor
In this project we are going to predict the stock price of any company.  I am using the LSTM model for this with the Tesla stock price as our data( collected by yahoo finance).


### About LSTM 
- **LSTM** *(From wikipedia)*

	**Long short-term memory** (**LSTM**) is an artificial [recurrent neural network](https://en.wikipedia.org/wiki/Recurrent_neural_network "Recurrent neural network") (RNN) architecture[[1]](https://en.wikipedia.org/wiki/Long_short-term_memory#cite_note-lstm1997-1) used in the field of [deep learning](https://en.wikipedia.org/wiki/Deep_learning "Deep learning"). Unlike standard [feedforward neural networks](https://en.wikipedia.org/wiki/Feedforward_neural_network "Feedforward neural network"), LSTM has feedback connections that make it a "general purpose computer" (that is, it can compute anything that a [Turing machine](https://en.wikipedia.org/wiki/Turing_machine "Turing machine") can).[[2]](https://en.wikipedia.org/wiki/Long_short-term_memory#cite_note-Siegelmann92-2) It can not only process single data points (such as images), but also entire sequences of data (such as speech or video). For example, LSTM is applicable to tasks such as unsegmented, connected [handwriting recognition](https://en.wikipedia.org/wiki/Handwriting_recognition "Handwriting recognition")[[3]](https://en.wikipedia.org/wiki/Long_short-term_memory#cite_note-3) or [speech recognition](https://en.wikipedia.org/wiki/Speech_recognition "Speech recognition").[[4]](https://en.wikipedia.org/wiki/Long_short-term_memory#cite_note-sak2014-4)[[5]](https://en.wikipedia.org/wiki/Long_short-term_memory#cite_note-liwu2015-5) [Bloomberg Business Week](https://en.wikipedia.org/wiki/Bloomberg_Business_Week "Bloomberg Business Week") wrote: "These powers make LSTM arguably the most commercial AI achievement, used for everything from predicting diseases to composing music."[[6]](https://en.wikipedia.org/wiki/Long_short-term_memory#cite_note-bloomberg2018-6)

	A common LSTM unit is composed of a **cell**, an **input gate**, an **output gate** and a **forget gate**. The cell remembers values over arbitrary time intervals and the three _gates_ regulate the flow of information into and out of the cell.

	LSTM networks are well-suited to [classifying](https://en.wikipedia.org/wiki/Classification_in_machine_learning "Classification in machine learning"), [processing](https://en.wikipedia.org/wiki/Computer_data_processing "Computer data processing") and [making predictions](https://en.wikipedia.org/wiki/Predict "Predict") based on [time series](https://en.wikipedia.org/wiki/Time_series "Time series") data, since there can be lags of unknown duration between important events in a time series. LSTMs were developed to deal with the exploding and [vanishing](https://en.wikipedia.org/wiki/Vanishing_gradient_problem "Vanishing gradient problem") gradient problems that can be encountered when training traditional RNNs. Relative insensitivity to gap length is an advantage of LSTM over RNNs, [hidden Markov models](https://en.wikipedia.org/wiki/Hidden_Markov_models "Hidden Markov models") and other sequence learning methods in numerous applications

## Requirements:
- numpy
- pandas
- seaborn
- matplotlib
- yfinance (yahoo finance)
- sklearn
- keras with tensorflow as backend
- GPU availability is a big plus.

### Steps to run it on your system:

**Note:** I am using Ubuntu 18.04 with anaconda environment and python-3.6.8

1. Get this project to your local system

2. Change directory to current project
	>cd stock_price_predictor

3. Create virtual environment ***[Optional]***  
Using Anaconda here( You may use python venv)
**Note:** Use tensorflow as backend in keras
- Use the terminal or an Anaconda Prompt for the following steps:

	> conda create -n myenv python=3.6.8

- Activate the new environment:

	> conda activate myenv

4. Run the python file

	> python main.py


	**Note:** If you have created a virtual environment , you may leave it by running
	>conda deactivate
### Interface
After running the python file you should be able to see the Mean-Absolute-Error of our model and a plot showing predicted and real stock prices.

**Note:** You can also refer the notebook file for better understanding.

