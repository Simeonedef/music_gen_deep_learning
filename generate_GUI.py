# coding=utf-8

import pickle
import random
import numpy
from music21 import instrument, note, stream, chord
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation

"""
EXAMPLE 2
Game menu with 3 difficulty options.
The MIT License (MIT)
Copyright 2017-2018 Pablo Pizarro R. @ppizarror
Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software
is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

# Import pygame and libraries
from pygame.locals import *
from random import randrange
import os
import pygame

# Import pygameMenu
import pygameMenu
from pygameMenu.locals import *

ABOUT = ["A small program to generate music using deep learning",
         "Author: Simeone de Fremond",
         PYGAMEMENU_TEXT_NEWLINE,
         "University of Manchester Third Year Project 2018-2019"]
COLOR_BACKGROUND = (252, 252, 252)
COLOR_TITLE = (107, 205, 255)
COLOR_SELECTED = (45, 160, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_WHITE = (255, 255, 255)
FPS = 60.0
MENU_BACKGROUND_COLOR = (250, 250, 250)
WINDOW_SIZE = (640, 480)

current_pos = None

# -----------------------------------------------------------------------------
# Init pygame
pygame.init()
os.environ['SDL_VIDEO_CENTERED'] = '1'

# Create pygame screen and objects
surface = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption('Music Generation using Deep Learning')
clock = pygame.time.Clock()
dt = 1 / FPS

# Global variables
HAS_BEEN_GEN = False
HAS_BEEN_GEN_RAND = False


# -----------------------------------------------------------------------------

def generate():
    # load the notes used to train the model
    print("Loading the notes used to train the model")
    with open('data/notes', 'rb') as filepath:
        notes = pickle.load(filepath)

    # Get all pitch names
    pitchnames = sorted(set(item for item in notes))
    # Get all pitch names
    n_vocab = len(set(notes))

    print("Getting network input")
    network_input, normalized_input = prepare_sequences(notes, pitchnames, n_vocab)
    print("Creating model")
    model = create_network(normalized_input, n_vocab)
    print("Generating notes")
    prediction_output = generate_notes(model, network_input, pitchnames, n_vocab)
    print("Creating midi file")
    create_midi(prediction_output)


def prepare_sequences(notes, pitchnames, n_vocab):
    # map between notes and integers and back
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    sequence_length = 100
    network_input = []
    # output = []
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        # sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        # output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    normalized_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    # normalize input
    normalized_input = normalized_input / float(n_vocab)

    return (network_input, normalized_input)


def create_network(network_input, n_vocab):
    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        return_sequences=True
    ))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    # Load the weights to each node
    model.load_weights('weights.hdf5')

    return model


def generate_notes(model, network_input, pitchnames, n_vocab):
    # pick a random sequence from the input as a starting point for the prediction
    start = numpy.random.randint(0, len(network_input)-1)

    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

    pattern = network_input[start]
    prediction_output = []

    # generate 500 notes
    for note_index in range(500):
        prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)

        index = numpy.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)

        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    return prediction_output


def generate_random():
    # load the notes used to train the model
    print("Loading the notes used to train the model")
    with open('data/notes', 'rb') as filepath:
        notes = pickle.load(filepath)

    # Get all pitch names
    pitchnames = sorted(set(item for item in notes))
    # Get all pitch names
    n_vocab = len(set(notes))

    print("Generating notes")
    prediction_output = generate_notes_random(pitchnames, n_vocab)
    print("Creating midi file")
    create_midi(prediction_output)


def generate_notes_random(pitchnames, n_vocab):
    # pick a random sequence from the input as a starting point for the prediction

    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

    prediction_output = []

    # generate 500 notes
    for note_index in range(500):
        index = random.randrange(0, n_vocab-1, 1)
        result = int_to_note[index]
        prediction_output.append(result)

    return prediction_output


def create_midi(prediction_output):
    offset = 0
    output_notes = []

    print(prediction_output)

    # create note and chord objects based on the values generated by the model
    for pattern in prediction_output:
        # pattern is a chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        elif pattern == 'rest':
            new_note = note.Rest()
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
        # pattern is a note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        # increase offset each iteration so that notes do not stack
        offset += 0.25

    midi_stream = stream.Stream(output_notes)

    midi_stream.write('midi', fp='test_output.mid')


# -----------------------------------------------------------------------------

def random_color():
    return randrange(0, 255), randrange(0, 255), randrange(0, 255)


def play_function(font):

    f = font.render('Generating music', 1, COLOR_WHITE)

    # Draw random color and text
    bg_color = random_color()
    f_width = f.get_size()[0]

    # Reset main menu and disable
    # You also can set another menu, like a 'pause menu', or just use the same
    # main_menu as the menu that will check all your input.
    main_menu.disable()

    # Continue playing
    surface.fill(bg_color)
    surface.blit(f, ((WINDOW_SIZE[0] - f_width) / 2, WINDOW_SIZE[1] / 2))
    pygame.display.flip()

    generate()

    global HAS_BEEN_GEN
    HAS_BEEN_GEN = True

    play_menu.enable()


def play_random_function(font):
    f = font.render('Generating random music', 1, COLOR_WHITE)

    # Draw random color and text
    bg_color = random_color()
    f_width = f.get_size()[0]

    # Reset main menu and disable
    # You also can set another menu, like a 'pause menu', or just use the same
    # main_menu as the menu that will check all your input.
    main_menu.disable()

    # Continue playing
    surface.fill(bg_color)
    surface.blit(f, ((WINDOW_SIZE[0] - f_width) / 2, WINDOW_SIZE[1] / 2))
    pygame.display.flip()

    generate_random()

    global HAS_BEEN_GEN_RAND
    HAS_BEEN_GEN_RAND = True

    play_menu.enable()


def music_function(font):
    if HAS_BEEN_GEN:
        f = font.render('Playing music', 2, COLOR_WHITE)
        help_text = font.render('Press esc to stop playing and go back', 1, COLOR_WHITE)
        # help_text2 = font.render('Press space to pause/unpause', 1, COLOR_WHITE)

        # Draw random color and text
        bg_color = random_color()
        f_width = f.get_size()[0]
        help_width = help_text.get_size()[0]
        # help2_width = help_text2.get_size()[0]
        f_height = f.get_size()[1]

        # Reset main menu and disable
        # You also can set another menu, like a 'pause menu', or just use the same
        # main_menu as the menu that will check all your input.
        main_menu.disable()

        # Continue playing
        surface.fill(bg_color)
        surface.blit(f, ((WINDOW_SIZE[0] - f_width) / 2, WINDOW_SIZE[1] / 2))
        surface.blit(help_text, ((WINDOW_SIZE[0] - help_width) / 2, (WINDOW_SIZE[1] / 2) + f_height + 2))
        # surface.blit(help_text2, ((WINDOW_SIZE[0] - help2_width) / 2, (WINDOW_SIZE[1] / 2) + 2*f_height + 4))
        pygame.display.flip()

        pygame.mixer.music.load('test_output.mid')
        pygame.mixer.music.play(0)

        # playing = True

        while True:
            # Application events
            playevents = pygame.event.get()
            for e in playevents:
                if e.type == QUIT:
                    exit()
                elif e.type == KEYDOWN:
                    if e.key == K_ESCAPE and main_menu.is_disabled():
                        pygame.mixer.music.stop()
                        main_menu.enable()
                        # Quit this function, then skip to loop of main-menu on line 217
                        return
                    # elif e.key == K_SPACE:
                    #     print("Trying to pause")
                    #     if playing:
                    #         print("Music is currently playing")
                    #         global current_pos
                    #         current_pos = pygame.mixer.music.get_pos()
                    #         print(current_pos)
                    #         pygame.mixer.music.stop()
                    #         playing = False
                    #     else:
                    #         print("No music playing")
                    #         print(current_pos)
                    #         pygame.mixer.music.set_pos(current_pos)
                    #         # pygame.mixer.music.play(0)
                    #         playing = True

            # Pass events to main_menu
            main_menu.mainloop(playevents)

        play_menu.enable()
    else:
        f = font.render('You need to generate some music first!', 1, COLOR_WHITE)
        help_text = font.render('Press any key to go back', 1, COLOR_WHITE)

        # Draw random color and text
        bg_color = random_color()
        f_width = f.get_size()[0]

        # Reset main menu and disable
        # You also can set another menu, like a 'pause menu', or just use the same
        # main_menu as the menu that will check all your input.
        main_menu.disable()

        # Continue playing
        surface.fill(bg_color)
        surface.blit(f, ((WINDOW_SIZE[0] - f_width) / 2, WINDOW_SIZE[1] / 2))
        surface.blit(help_text, ((WINDOW_SIZE[0] - f_width) / 2, WINDOW_SIZE[1] / 2 - f_width))
        pygame.display.flip()

        while True:
            # Application events
            playevents = pygame.event.get()
            for e in playevents:
                if e.type == KEYDOWN:
                    main_menu.enable()
                    return

            # Pass events to main_menu
            main_menu.mainloop(playevents)

        play_menu.enable()


def music_random_function(font):
    if HAS_BEEN_GEN_RAND:
        f = font.render('Playing music', 2, COLOR_WHITE)
        help_text = font.render('Press esc to stop playing and go back', 1, COLOR_WHITE)
        # help_text2 = font.render('Press space to pause/unpause', 1, COLOR_WHITE)

        # Draw random color and text
        bg_color = random_color()
        f_width = f.get_size()[0]
        help_width = help_text.get_size()[0]
        # help2_width = help_text2.get_size()[0]
        f_height = f.get_size()[1]

        # Reset main menu and disable
        # You also can set another menu, like a 'pause menu', or just use the same
        # main_menu as the menu that will check all your input.
        main_menu.disable()

        # Continue playing
        surface.fill(bg_color)
        surface.blit(f, ((WINDOW_SIZE[0] - f_width) / 2, WINDOW_SIZE[1] / 2))
        surface.blit(help_text, ((WINDOW_SIZE[0] - help_width) / 2, (WINDOW_SIZE[1] / 2) + f_height + 2))
        # surface.blit(help_text2, ((WINDOW_SIZE[0] - help2_width) / 2, (WINDOW_SIZE[1] / 2) + 2*f_height + 4))
        pygame.display.flip()

        pygame.mixer.music.load('test_output_random.mid')
        pygame.mixer.music.play(1)

        # playing = True

        while True:
            # Application events
            playevents = pygame.event.get()
            for e in playevents:
                if e.type == QUIT:
                    exit()
                elif e.type == KEYDOWN:
                    if e.key == K_ESCAPE and main_menu.is_disabled():
                        pygame.mixer.music.stop()
                        main_menu.enable()
                        # Quit this function, then skip to loop of main-menu on line 217
                        return
                    # elif e.key == K_SPACE:
                    #     print("Trying to pause")
                    #     if playing:
                    #         print("Music is currently playing")
                    #         global current_pos
                    #         current_pos = pygame.mixer.music.get_pos()
                    #         print(current_pos)
                    #         pygame.mixer.music.stop()
                    #         playing = False
                    #     else:
                    #         print("No music playing")
                    #         print(current_pos)
                    #         pygame.mixer.music.set_pos(current_pos)
                    #         # pygame.mixer.music.play(0)
                    #         playing = True

            # Pass events to main_menu
            main_menu.mainloop(playevents)

        play_menu.enable()
    else:
        f = font.render('You need to generate some random music first!', 1, COLOR_WHITE)
        help_text = font.render('Press any key to go back', 1, COLOR_WHITE)

        # Draw random color and text
        bg_color = random_color()
        f_width = f.get_size()[0]

        # Reset main menu and disable
        # You also can set another menu, like a 'pause menu', or just use the same
        # main_menu as the menu that will check all your input.
        main_menu.disable()

        # Continue playing
        surface.fill(bg_color)
        surface.blit(f, ((WINDOW_SIZE[0] - f_width) / 2, WINDOW_SIZE[1] / 2))
        surface.blit(help_text, ((WINDOW_SIZE[0] - f_width) / 2, WINDOW_SIZE[1] / 2 - f_width))
        pygame.display.flip()

        while True:
            # Application events
            playevents = pygame.event.get()
            for e in playevents:
                if e.type == KEYDOWN:
                    main_menu.enable()
                    return

            # Pass events to main_menu
            main_menu.mainloop(playevents)

        random_menu.enable()

def main_background():
    surface.fill(COLOR_BACKGROUND)


# -----------------------------------------------------------------------------
# PLAY MENU
play_menu = pygameMenu.Menu(surface,
                            bgfun=main_background,
                            color_selected=COLOR_SELECTED,
                            font=pygameMenu.fonts.FONT_BEBAS,
                            font_color=COLOR_BLACK,
                            font_size=30,
                            menu_alpha=100,
                            menu_color=MENU_BACKGROUND_COLOR,
                            menu_color_title=COLOR_TITLE,
                            menu_height=int(WINDOW_SIZE[1]),
                            menu_width=int(WINDOW_SIZE[0]),
                            onclose=PYGAME_MENU_DISABLE_CLOSE,
                            option_shadow=False,
                            title='Music menu',
                            window_height=WINDOW_SIZE[1],
                            window_width=WINDOW_SIZE[0]
                            )

play_menu.add_option('Generate music', play_function,
                     pygame.font.Font(pygameMenu.fonts.FONT_FRANCHISE, 30))

play_menu.add_option('Play music', music_function,
                     pygame.font.Font(pygameMenu.fonts.FONT_FRANCHISE, 30))

play_menu.add_option('Return to main menu', PYGAME_MENU_BACK)


random_menu = pygameMenu.Menu(surface,
                              bgfun=main_background,
                              color_selected=COLOR_SELECTED,
                              font=pygameMenu.fonts.FONT_BEBAS,
                              font_color=COLOR_BLACK,
                              font_size=30,
                              menu_alpha=100,
                              menu_color=MENU_BACKGROUND_COLOR,
                              menu_color_title=COLOR_TITLE,
                              menu_height=int(WINDOW_SIZE[1]),
                              menu_width=int(WINDOW_SIZE[0]),
                              onclose=PYGAME_MENU_DISABLE_CLOSE,
                              option_shadow=False,
                              title='Random Music menu',
                              window_height=WINDOW_SIZE[1],
                              window_width=WINDOW_SIZE[0]
                              )

random_menu.add_option('Generate music', play_random_function,
                       pygame.font.Font(pygameMenu.fonts.FONT_FRANCHISE, 30))

random_menu.add_option('Play music', music_random_function,
                       pygame.font.Font(pygameMenu.fonts.FONT_FRANCHISE, 30))

random_menu.add_option('Return to main menu', PYGAME_MENU_BACK)

# ABOUT MENU
about_menu = pygameMenu.TextMenu(surface,
                                 bgfun=main_background,
                                 color_selected=COLOR_SELECTED,
                                 font=pygameMenu.fonts.FONT_FRANCHISE,
                                 font_color=COLOR_BLACK,
                                 # font_size_title=30,
                                 font_title=pygameMenu.fonts.FONT_BEBAS,
                                 menu_color=MENU_BACKGROUND_COLOR,
                                 menu_color_title=COLOR_TITLE,
                                 menu_height=int(WINDOW_SIZE[1]),
                                 menu_width=int(WINDOW_SIZE[0]),
                                 onclose=PYGAME_MENU_DISABLE_CLOSE,
                                 option_shadow=False,
                                 text_color=COLOR_BLACK,
                                 text_fontsize=20,
                                 title='About',
                                 window_height=WINDOW_SIZE[1],
                                 window_width=WINDOW_SIZE[0]
                                 )
for m in ABOUT:
    about_menu.add_line(m)
about_menu.add_line(PYGAMEMENU_TEXT_NEWLINE)
about_menu.add_option('Return to menu', PYGAME_MENU_BACK)

# MAIN MENU
main_menu = pygameMenu.Menu(surface,
                            bgfun=main_background,
                            color_selected=COLOR_SELECTED,
                            font=pygameMenu.fonts.FONT_BEBAS,
                            font_color=COLOR_BLACK,
                            font_size=30,
                            menu_alpha=100,
                            menu_color=MENU_BACKGROUND_COLOR,
                            menu_color_title=COLOR_TITLE,
                            menu_height=int(WINDOW_SIZE[1]),
                            menu_width=int(WINDOW_SIZE[0]),
                            onclose=PYGAME_MENU_DISABLE_CLOSE,
                            option_shadow=False,
                            title='Music Generation',
                            window_height=WINDOW_SIZE[1],
                            window_width=WINDOW_SIZE[0]
                            )
main_menu.add_option('Music', play_menu)
main_menu.add_option('Random', random_menu)
main_menu.add_option('About', about_menu)
main_menu.add_option('Quit', PYGAME_MENU_EXIT)

# -----------------------------------------------------------------------------
# Main loop
while True:

    # Tick
    clock.tick(60)

    # Application events
    events = pygame.event.get()
    for event in events:
        if event.type == QUIT:
            exit()

    # Main menu
    main_menu.mainloop(events)

    # Flip surface
    pygame.display.flip()