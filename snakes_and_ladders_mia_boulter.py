'''
Statistical Analysis of Snakes and Ladders
Year 3 Numerical Modelling of Physical Systems
By Mia Boulter

'''

import time
import random
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

class Dice():
    """ A fair dice, with the capability to give a constant roll 
    
    Public Methods
    --------------
    roll
        roll the dice returning the outcome

    Attributes
    ----------
    sides : int
        number of sides of the dice
    constant_roll : False or int
        False, or an integer which is constantly returned when the dice is rolled.
    """

    def __init__(self, sides=6, constant_roll=False):
        """ Costruct a dice
        
        Parameters
        ----------
        sides : int
            Number of sides the dice should have.
        constant_roll : bool, optional
            If set, the dice will always return this number when the roll method is called. 
        """
        self.sides = sides
        self.constant_roll = constant_roll

    def roll(self):
        """ Roll the dice. 
        
        Returns
        -------
        roll : int
            A random number from 1 up to the number of sides.
            Or a predefined value if constant roll is set.
        """
        if self.constant_roll:
            return self.constant_roll
        return random.randint(1, self.sides)

class Rules():
    """ Rules for snakes and ladders. """

    def __init__(self, bonus_roll=False, bonus_roll_first=False, exact_finish=True, reverse_overshoot=False):
        '''
        Parameters
        ----------
        bonus_roll : 'single', 'multiple' or False, optional (default=False)
            If not False, if the player gets the maximum possibility in the dice roll, they get another roll.
            If 'single', there is a limit of 1 bonus roll. If 'multiple' a person's bonus rolls are unlimited.
        bonus_roll_first : bool, optional (default=False)
            Only active if bonus_roll != False. If True, the ladders, snakes, or finish conditions are checked
            after the player gets their bonus roll, otherwise the bonus roll happens last.
        exact_finish : bool, optional (default=True)
            Whether the player needs to roll the exact number to land on the finishing square in order
            to finish. I False, they will be placed on the finishing square in such a case.
        reverse_overshoot : bool, optional (default=False)
            Only active if exact_finish == True. If True, should the player overshoot the finish, they move
            backwards by the number of squares equal to the distance to the finish subtracted from their roll. 
            If False, they don't move.
        '''
        self.bonus_roll = bonus_roll
        self.bonus_roll_first = bonus_roll_first
        self.exact_finish = exact_finish
        self.reverse_overshoot = reverse_overshoot

class Board():
    """ A snakes and ladders board.

    Public Methods
    --------------
    board_map
        Generate a dict containing the board's information
    transition_matrix
        Generate the board's transition matrix
    
    Class Methods
    -------------
    random
        Generate a random board"""

    def __init__(self, board_map):
        self.squares = board_map['squares']
        self.snakes = board_map['snakes']
        self.ladders = board_map['ladders']
        self.start_off_board = board_map['start_off_board']
        if self.start_off_board:
            self.start_square = 0
        else:
            self.start_square = 1
        self.final_square = self.squares
        self.transition_matrix_value = None

    def board_map(self):
        """ Return the board map, including any values that may have been changed
        since initialisation of the board. """
        return {'squares':self.squares, 'snakes': self.snakes, 'ladders': self.ladders, 'start_off_board': self.start_off_board}

    def __repr__(self):
        return f"Board({repr(self.board_map())})"

    @classmethod
    def random(cls, squares=100, start_off_board=True, snakes=9, ladders=9):
        """
        Create a random snakes and ladders board.

        Parameters
        ----------
        squares : int, optional (default=100)
            Number of squares on the board.
        start_off_board : bool, optional (default=True)
            Whether the players should start off the board. If False, players start on the
            first square and no ladders will start on this first square.
        snakes : int, optional (default=19)
            Number of snakes to put on the board.
        ladders : int, optional (default=19)
            Number of ladders to put on the board.
        """
        board_map = {'squares': squares, 'snakes':{}, 'ladders':{},
                     'start_off_board':start_off_board}

        if (snakes + ladders) > (squares / 2 - 4):
            raise ValueError("Too many snakes and ladders to fit on board")

        # NB. squares start at 0 and go to squares-1
        unused_squares = [i for i in range(squares)]
        if start_off_board:
            ladder_min_start = 1
        else:
            ladder_min_start = 2

        for _ in range(snakes):
            snake_head = 0
            while snake_head in (squares-1, 0, 1):
                snake_head = random.choice(unused_squares[1:]) # must be one unused below for tail
            snake_tail = 0
            while snake_tail == 0:
                snake_tail = random.choice(unused_squares[:unused_squares.index(snake_head)])
            unused_squares.remove(snake_tail)
            unused_squares.remove(snake_head)
            board_map['snakes'][snake_head] = snake_tail

        for _ in range(ladders):
            ladder_bottom = ladder_min_start - 1
            while ladder_bottom in (ladder_min_start-1, ladder_min_start-2, squares-1):
                ladder_bottom = random.choice(unused_squares[:-1])
            ladder_top = random.choice(unused_squares[unused_squares.index(ladder_bottom)+1:])
            unused_squares.remove(ladder_bottom)
            unused_squares.remove(ladder_top)
            board_map['ladders'][ladder_bottom] = ladder_top

        return cls(board_map)

    def transition_matrix(self, rules=Rules(), dice_sides=6):
        """ Generate a transition matrix for this board
        
        Parameters
        ----------
        rules : Rules instance, optional
            A rule set to use for generation. Uses the default rule set as a default.
        dice_sides : int, optional
            The number of sides the dice should have for generation. Defaults as 6.
            
        Returns
        -------
        transition matrix : TransitionMatrix instance
            The transition matrix of the board given the rules and dice sides
        """
        if rules.bonus_roll:
            raise ValueError("To calculate the average number of turns analytically, bonus_roll must be set to False")
        tm = []

        for square in range(self.squares+1):
            row = [0 for _ in range(self.squares + 1)]
            if square not in self.snakes and square not in self.ladders:
                for roll in range(1, dice_sides+1):
                    game = Game(['player_name'], self, rules=rules, dice=Dice(constant_roll=roll)) # default rules
                    game.move_player_to_position('player_name', square)
                    game.do_turn()
                    new_position = game.players['player_name'].position
                    #print("new:", new_position)
                    row[new_position] = row[new_position] + 1/dice_sides
            # sometimes what should be 1 is stored as 0.999999999
            # the following is to mitigate that
            for c, i in enumerate(row):
                if round(i, 15) == 1:
                    row[c] = 1
            tm.append(row)

        return TransitionMatrix(tm)

class TransitionMatrix(np.ndarray):
    """ Transition matrix. 
    
    Public Methods
    --------------
    state_is_transient
        Whether each state of the matrix is transient
    canonical_arrangement
        The transition matrix in canonical form
    transient
        The transient portion of the transition matrix
    fundamental_matrix
        The fundamental matrix for this
    """

    def __new__(cls, a):
        return np.asarray(a).view(cls)

    def state_is_transient(self):
        """ Return a list of the states in input order with whether they
        are transient (True) or absorbing (False)."""
        return_val = [True for _ in self[0]]
        for index, row in enumerate(self):
            if row[index] == 1:
                return_val[index] = False
        return return_val

    def canonical_arrangement(self):
        """ Return the transition matrix in canonical arrangement such that the transitional
        states come first, then the absorbing states.

        Returns
        -------
        canonical_matrix : TransitionMatrix
            The transition matrix in canonical arrangement.
        new_arrangement : list
            What position in the original matrix each position in the canonical
            arrangement corresponds to.
        """

        is_transient = self.state_is_transient()
        if all(is_transient):
            raise ValueError("Transition matrix must have at least one absorbing state.")

        old_arrangement = [i for i, _ in enumerate(self[0])]
        new_arrangement = [i for c, i in enumerate(old_arrangement) if is_transient[c]] \
                        + [i for c, i in enumerate(old_arrangement) if not is_transient[c]]
        new_rows = np.array([row[new_arrangement] for row in self])
        return self.__class__(new_rows[new_arrangement]), new_arrangement

    def transient(self):
        """ Return Q, the transient portion of the matrix from the canonical form.

        Returns
        -------
        Q : ndarray
            the transient portion of the matrix.
        arr_transient_only : list
            What each row/col in Q corresponds to in row in the original transition matrix.

        """
        can_arr, arr = self.canonical_arrangement()
        num_of_transient = sum(1 * can_arr.state_is_transient())
        Q = np.array([row[:num_of_transient] for row in can_arr[:num_of_transient]])
        arr_transient_only = arr[:num_of_transient]
        self.Q = Q
        self.arr_transient_only = arr_transient_only
        return Q, arr_transient_only

    def fundamental_matrix(self):
        """ Return the fundamental matrix. 
        
        Returns
        -------
        N : ndarray
            the fundamental matrix
        arr_transient_only : list
            What each row/col in N corresponds to in row in the original transition matrix.
        """
        Q, arr_transient_only = self.transient()
        I = np.identity(len(Q[0]))
        N = np.linalg.inv(I - Q)
        self.N = N
        return N, arr_transient_only

class Player:
    """ Data describing a player in a snakes and ladders game
    
    Attributes
    ----------
    name : str
        player's name
    position : int
        current square on the board that the player is on
    has_finished : bool
        whether the player has reached the final square
    finished_on_turn : False, int
        if the player has finished, what turn they finished on
    """

    def __init__(self, name, start_position):
        """ Initialise player
        
        Parameters
        ----------
        name : str
            the player's name
        start_position
            the square the player starts at on the board
        """
        self.name = name
        self.position = start_position
        self.has_finished = False
        self.finished_on_turn = False

class Game():
    """ A simulation of a game of snakes and ladders
    
    Attributes
    ----------
    board : Board instance
        the board being played on
    dice : Dice instance
        The dice being used
    player_finish_order : list
        Players who have finished in order of finishing
    game_is_over : bool
        Whether all players have arrived at the finish
    rules : Rules instance
        The rules being used
    turns_passed : int
        Current turn count
    players : dict
        The players names are keys, the corresponding Player object is value."""

    def __init__(self, player_names, board=Board.random(), dice=Dice(6), rules=Rules()):
        """ Initialise a simulation of snakes and ladders.
        player_names : iterable of strings
            Iterable which contains the names of players.
        board : Board instance, optional
            The board to be used. Defaults to default board
        dice : Dice instance, optional
            The dice to be used. Defaults to 6 sided fair dice.
        rules : Rules instance, optional
            The rules to be used. Defaults to default rules.
        """
        self.board = board
        self.dice = dice
        self.player_finish_order = []
        self.game_is_over = False
        self.rules = rules
        self.turns_passed = 0
        self._old_position = 0
        self._new_position = 0
        self._player = None

        self.players = {}
        for name in player_names:
            self.players[name] = Player(name, self.board.start_square)

    def do_turn(self):
        """ Do a turn in the game, progressing players accordingly."""
        if not self.game_is_over:
            self.turns_passed += 1
            #print(f"Turn {self.turns_passed}")

            for player in self.players.values():

                if not player.position == self.board.final_square:
                    self._player = player
                    #print(f"{player.name}'s Turn:")

                    self._old_position = player.position

                    roll = self.dice.roll()
                    self._new_position = self._old_position + roll
                    #print(f"    Rolled a {roll}\n    New position: {self._new_position}")

                    if self.rules.bonus_roll and roll == self.dice.sides:

                        # Check if bonus roll shoult be done before other things
                        if self.rules.bonus_roll_first:

                            if self.rules.bonus_roll == 'single':
                                roll = self.dice.roll()
                                self._new_position = self._old_position + roll

                            else: # infinite bonus rolls
                                while roll == self.dice.sides:
                                    roll = self.dice.roll()
                                    self._new_position = self._old_position + roll

                            self._do_square_considerations()

                        else: # bonus roll happens after other stuff

                            if self.rules.bonus_roll == 'single':
                                self._do_square_considerations()
                                roll = self.dice.roll()
                                self._new_position = self._old_position + roll
                                self._do_square_considerations()

                            else:
                                self._do_square_considerations()
                                while roll == self.dice.sides:
                                    roll = self.dice.roll()
                                    self._new_position = self._old_position + roll
                                    self._do_square_considerations()
                    else:
                        self._do_square_considerations()

    def _do_square_considerations(self):
        """ On the square that the player is on, do the considerations that need to be done
        to calculate the new position for the player. """

        # Check if has overshot finish square
        # it is important that this comes before snake/ladder checks to
        # stop a player from landing and staying on a snake head / ladder bottom
        if self._new_position > self.board.final_square:
            if self.rules.exact_finish:
                if self.rules.reverse_overshoot:
                    self._new_position = 2*self.board.final_square - self._new_position
                else:
                    self._new_position = self._old_position
                    #print("    Has overshot finish, stayed in old position.")
            else:
                self._new_position = self.board.final_square

        # Check if hit a snake
        if self._new_position in self.board.snakes:
            self._new_position = self.board.snakes[self._new_position]
            #print(f"    Hit a snake!\n    Moved down to square {self._new_position}")

        # Check if hit a ladder
        elif self._new_position in self.board.ladders:
            self._new_position = self.board.ladders[self._new_position]
            #print(f"    Hit a ladder!\n    Moved up to square {self._new_position}")

        # Check if has hit finish square
        if self._new_position == self.board.final_square:
            self.player_finish_order.append(self._player)
            self._player.has_finished = True
            self._player.finished_on_turn = self.turns_passed
            #print("    Player has Finished!")
        self._player.position = self._new_position

        # Check if game is over
        if len(self.player_finish_order) == len(self.players):
            self.game_is_over = True

    def move_player_to_position(self, player, square):
        """ Place the player on a certain square on the board.
        
        Parameters
        ----------
        player : str
            Name of player (must preexist in the game)
        square : int
            Square on the board to place the player onto.
        """
        self.players[player].position = square

def ms_word_matrix(arr):
    """ Generate a unicode format matrix that can be put straight into microsoft word
    equation. Then change the display mode to professional to see it.

    Parameters
    ----------
    arr : numpy.ndarray

    Returns
    -------
    string : str
        Unicode format string for displaying the matrix in msword
    """
    start = '[■('
    rows_str = ["&".join(f'{i:.3f}' for i in row) for row in arr]
    end = ')]'
    return start + '@'.join(rows_str) + end

def avg_turns_numeric(number_of_games, board, dice_sides=6, rules=Rules()):
    ''' Find the average number of turns by running the game number_of_games times.

    Print and return the results.
    
    Parameters
    ----------
    number_of_games : int
        Number of times to run the game.
    board : Board
        The board to use
    dice_sides : int, optional
        Number of sides for the dice to have. Default 6.
    rules : Rules, optional
        Rules to use. Default to default rules.

    Returns
    -------
    avg : float
        Average turns to complete the game from start square of board.
    variance : float
        The variance on this average value.
    turn_array : list
        List of number of turns each player took.

    '''
    tic = time.perf_counter()
    G = Game([str(n) for n in range(number_of_games)], board, rules=rules, dice=Dice(dice_sides))
    while not G.game_is_over:
        G.do_turn()
    turn_array = [player.finished_on_turn for player in G.players.values()]
    avg = sum(turn_array) / len(G.players)
    variance = sum([(player.finished_on_turn - avg)**2 for player in G.players.values()]) / len(G.players)
    toc = time.perf_counter()
    print("-------------------------------------------------\n"\
          "Numeric Results For Average Turn to Complete Game\n"\
          "-------------------------------------------------")
    print("           Answer:", avg)
    print("         Variance:", variance)
    print("No. games sampled:", number_of_games)
    print("       Time taken:", toc-tic, "seconds\n-------------------------------------------------")
    return avg, variance, turn_array

def avg_turns_numeric_plot(number_of_games, board, dice_sides=6, rules=Rules()):
    ''' Plot an array of number of players against number of turns.
    
    Run the game a specified number of times to obtain this.
    
    Plot shows the statistical likelihood of players to finish on each turn number.

    Also use the transition matrix of the board to predict the number of turns, and 
    plot on the graph.

    Parameters
    ----------
    number_of_games : int
        Number of times to run the game.
    board : Board
        The board to use
    dice_sides : int, optional
        Number of sides for the dice to have. Default 6.
    rules : Rules, optional
        Rules to use. Default to default rules.
    
    '''
    # numeric stuff
    avg, var, turn_array = avg_turns_numeric(number_of_games, board, dice_sides, rules)
    turn_dict = {}
    for i in turn_array:
        if i not in turn_dict:
            turn_dict[i] = 1
        else:
            turn_dict[i] += 1

    x = turn_dict.keys() # number of turns
    y = turn_dict.values() # number of players

    # analytic stuff
    P = board.transition_matrix(rules=rules, dice_sides=dice_sides)
    Q, _ = P.transient()
    x2 = [i for i in range(max(x))]
    y2 = np.diff([1-sum(np.linalg.matrix_power(Q, i)[0]) for i in x2]) * number_of_games
    x2 = x2[1:]

    plt.bar(x, y, label="Numeric Values")
    plt.plot(x2, y2, color='r', alpha=0.9, label='Markov prediction')
    plt.xlabel("Number of turns")
    plt.ylabel("Number of players")
    plt.legend(loc='best')
    plt.show()

def plot_entropy(max_turns, board, dice_sides=6, rules=Rules()):
    """ For a board, use the transition matrix to find the
    expected entropy after each turn.

    Do this up to a turn limit, max_turns, then plot the results.

    Parameters
    ----------
    max_turns : int
        The number of turns to plot up to.
    board : Board
        The board to use
    dice_sides : int, optional
        Number of sides for the dice to have. Default 6.
    rules : Rules, optional
        Rules to use. Default to default rules.
    """
    P = board.transition_matrix(rules=rules, dice_sides=dice_sides)
    x = [i for i in range(max_turns)]
    y = [stats.entropy(np.linalg.matrix_power(P, i)[0]) for i in x]
    print("Max Entropy at turn:", np.argmax(y))
    print("Max Entropy value:", max(y))
    plt.plot(x, y)
    plt.xlabel('Number of Turns Passed')
    plt.ylabel("Entropy")
    plt.grid()
    plt.show()

def plot_entropy_comparison(max_turns, board, dice_sides=6):
    """For a board, use the transition matrix to find the
    expected entropy after each turn.

    Do this up to a turn limit, max_turns, then plot the results.

    Comparison of three finish conditions
    - non-exact finish
    - exact finish
    - reverse overshoot

    Parameters
    ----------
    max_turns : int
        The number of turns to plot up to.
    board : Board
        The board to use
    dice_sides : int, optional
        Number of sides for the dice to have. Default 6.
    """

    # exact_finish == True
    rules = Rules(exact_finish=True)
    P = board.transition_matrix(rules=rules, dice_sides=dice_sides)
    x = [i for i in range(max_turns)]
    y = [stats.entropy(np.linalg.matrix_power(P, i)[0]) for i in x]
    print("Max Entropy at turn (exact_finish == True):", np.argmax(y))
    print("Max Entropy value:", max(y))
    plt.plot(x, y, label="Rules(exact_finish=True)")

    # reverse_overshoot == True
    rules = Rules(exact_finish=True, reverse_overshoot=True)
    P = board.transition_matrix(rules=rules, dice_sides=dice_sides)
    x = [i for i in range(max_turns)]
    y = [stats.entropy(np.linalg.matrix_power(P, i)[0]) for i in x]
    print("Max Entropy at turn (exact_finish=True, reverse_overshoot=True):", np.argmax(y))
    print("Max Entropy value:", max(y))
    plt.plot(x, y, label="Rules(exact_finish=True, reverse_overshoot=True)", color='#A72432')

    # exact_finish == False
    rules = Rules(exact_finish=False)
    P = board.transition_matrix(rules=rules, dice_sides=dice_sides)
    x = [i for i in range(max_turns)]
    y = [stats.entropy(np.linalg.matrix_power(P, i)[0]) for i in x]
    print("Max Entropy at turn (exact_finish=False):", np.argmax(y))
    print("Max Entropy value:", max(y))
    plt.plot(x, y, label="Rules(exact_finish=False)", color='#008542')

    # complete plotting

    plt.xlabel('Number of Turns Passed')
    plt.ylabel("Entropy")
    plt.grid()
    plt.legend(loc='best')
    plt.show()

def avg_turns_analytic(board, dice_sides=6, rules=Rules()):
    """ Get the average turns taken to complete a game.

    Do this using the transition matrix of the board.

    Print and return the results.

    Parameters
    ----------
    board : Board
        The board to use
    dice_sides : int, optional
        Number of sides for the dice to have. Default 6.
    rules : Rules, optional
        Rules to use. Default to default rules.

    Returns
    -------
    t : list
        Average turns to finish by start positions in arr_transient_only.
    avg_dict : dict
        Square mapping to average turns to finish
    variance : list
        The variances corresponding to t.
    var_dict : dict
        The variances corresponding to avg_dict
    N : ndarray
        Fundamental matrix of the board
    a_valid : list
        Valid squares on the board that the player can be on.
    arr_transient_only : list
        What each row/col in N or t corresponds to in row in the original transition matrix.
    """
    tic = time.perf_counter()

    P = board.transition_matrix(rules=rules, dice_sides=dice_sides)
    N, arr_transient_only = P.fundamental_matrix()
    c = np.array([1 for _ in N[0]])

    t = N@c
    variance = (2*N - np.identity(len(t))) @ t - t*t

    avg_dict = {}
    var_dict = {}
    a_valid = []
    for index, square in enumerate(arr_transient_only):
        if square not in board.snakes and square not in board.ladders:
            avg_dict[square] = t[index]
            var_dict[square] = variance[index]
            a_valid.append(square)
    if board.start_square == 1:
        del avg_dict[0]
        del var_dict[0]

    ans_specific = avg_dict[board.start_square]
    var_specific = var_dict[board.start_square]
    toc = time.perf_counter()

    print("-------------------------------------\n"\
          "Analytic Average Turns to Finish Game\n"\
          "-------------------------------------")
    print("    Answer:", ans_specific)
    print("  Variance:", var_specific)
    print("Time taken:", toc-tic, "seconds")
    print("-------------------------------------\n")

    return t, avg_dict, variance, var_dict, N, a_valid, arr_transient_only

def plot_t_with_variance(board, dice_sides=6, rules=Rules()):
    """ On separate subplots, plot the average turns to finish, and the variances
    against the square on the board.

    Parameters
    ----------
    board : Board
        The board to use
    dice_sides : int, optional
        Number of sides for the dice to have. Default 6.
    rules : Rules, optional
        Rules to use. Default to default rules.
    """

    _, avg_dict, _, var_dict, _, _, _ = avg_turns_analytic(board, dice_sides, rules)

    x = avg_dict.keys()
    y = avg_dict.values()

    plt.subplot(2, 1, 1)
    plt.bar(x, y)

    plt.xlabel('Square on board')
    plt.ylabel('Average turns to absorption')
    plt.title("Average turns to finish by square")

    y = var_dict.values()

    plt.subplot(2, 1, 2)
    plt.bar(x, y, color='#A72432')
    plt.title('Variances on average turns to finish by square')

    plt.xlabel('Square on board')
    plt.ylabel('Variance')

    plt.tight_layout()
    plt.show()

def plot_expected_occupation_frequency(board, dice_sides=6):
    """ Plot expected number of times each square will be occupied before
    the player finishes.

    Comparison of three finish conditions
    - non-exact finish
    - exact finish
    - reverse overshoot

    Parameters
    ----------
    board : Board
        The board to use
    dice_sides : int, optional
        Number of sides for the dice to have. Default 6.
    """
    width = 0.2 # width of bars in plots

    # exact_finish == True
    rules = Rules(exact_finish=True)
    t, a_dict, variance, v_dict, N, a_valid, a_trans = avg_turns_analytic(board, dice_sides, rules)
    x = np.array(a_valid)
    y1 = [N[0][c] for c, i in enumerate(a_trans) if i in a_valid]
    plt.bar(x-width, y1, width=width, align='center', label="Rules(exact_finish=True)")

    # reverse_overshoot == True
    rules = Rules(exact_finish=True, reverse_overshoot=True)
    t, a_dict, variance, v_dict, N, a_valid, a_trans = avg_turns_analytic(board, dice_sides, rules)
    x = np.array(a_valid)
    y2 = [N[0][c] for c, i in enumerate(a_trans) if i in a_valid]
    plt.bar(x, y2, width=width, align='center', label="Rules(exact_finish=True, reverse_overshoot=True)", color='#A72432')

    # exact_finish == False
    rules = Rules(exact_finish=False)
    t, a_dict, variance, v_dict, N, a_valid, a_trans = avg_turns_analytic(board, dice_sides, rules)
    x = np.array(a_valid)
    y3 = [N[0][c] for c, i in enumerate(a_trans) if i in a_valid]
    plt.bar(x+width, y3, width=width, align='center', label="Rules(exact_finish=False)", color='#008542')

    # finish off plot

    plt.ylabel("Expected number of times occupied before absorption")
    plt.xlabel("Square on board")
    plt.legend(loc="best")

    # reverse_overshoot
    plt.show()

def plot_prob_of_absorption_by_turn(max_turns, board, dice_sides=6):
    """ Plots of how the cumulative probability of finishing varies with turn.

    Get the cumulative probabilities up to a turn limit, max_turns, then plot the results.

    Comparison of three finish conditions
    - non-exact finish
    - exact finish
    - reverse overshoot

    Parameters
    ----------
    max_turns : int
        The number of turns to plot up to.
    board : Board
        The board to use
    dice_sides : int, optional
        Number of sides for the dice to have. Default 6.

    """

    # exact_finish == True
    rules = Rules(exact_finish=True)
    P = board.transition_matrix(rules=rules, dice_sides=dice_sides)
    Q, _ = P.transient()
    x = np.arange(0, max_turns, step=1)
    y = [1-sum(np.linalg.matrix_power(Q, i)[0]) for i in x]
    plt.plot(x, y, label="Rules(exact_finish=True)")

    # reverse_overshoot == True
    rules = Rules(exact_finish=True, reverse_overshoot=True)
    P = board.transition_matrix(rules=rules, dice_sides=dice_sides)
    Q, _ = P.transient()
    x = np.arange(0, max_turns, step=1)
    y = [1-sum(np.linalg.matrix_power(Q, i)[0]) for i in x]
    plt.plot(x, y, label="Rules(exact_finish=True, reverse_overshoot=True)", color='#A72432')

    # exact_finish == False
    rules = Rules(exact_finish=False)
    P = board.transition_matrix(rules=rules, dice_sides=dice_sides)
    Q, _ = P.transient()
    x = np.arange(0, max_turns, step=1)
    y = [1-sum(np.linalg.matrix_power(Q, i)[0]) for i in x]
    plt.plot(x, y, label="Rules(exact_finish=False)", color='#008542')

    # finish off plot

    plt.ylabel("Probability of absorption by turn")
    plt.xlabel("Turn")
    plt.legend(loc="best")
    plt.grid()
    
    # reverse_overshoot
    plt.show()

def main():
    """ Testing """

    #example_board = Board({'squares': 9, 'snakes': {8:2}, 'ladders': {4:7}, 'start_off_board': True})
    #example_board = Board.random()

    example_board = Board({'squares': 100,
                           'snakes': {97: 55, 68: 3, 5: 2, 25: 7, 78: 37, 79: 1, 14: 6, 45: 10, 39: 21},
                           'ladders': {88: 93, 15: 70, 74: 95, 67: 71, 84: 86, 24: 46, 28: 31, 34: 76},
                           'start_off_board': True})

    #avg_turns_numeric(1000, example_board)
    #avg_turns_analytic(example_board)

    avg_turns_numeric_plot(10000, example_board)
    #plot_t_with_variance(example_board)
    #plot_expected_occupation_frequency(example_board)
    #plot_prob_of_absorption_by_turn(300, example_board)
    #plot_entropy_comparison(300, example_board)

if __name__ == "__main__":
    main()
