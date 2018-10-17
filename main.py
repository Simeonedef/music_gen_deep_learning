from music21 import *
import glob

notes_to_parse = None

notes = [] #for testing printing
pitches = []

offset = True
rest = False
offset_normalization = True
offset_rounding = 3

for file in glob.glob("midi/*.mid"):
    #file = "midi/Wtcii01a.mid"
    midi = converter.parse(file)
    print("Parsing %s" % file)

    highest_offset = 0.0

    try: # file has instrument parts
        inst = instrument.partitionByInstrument(midi)
        print("Number of instrument parts: " + str(len(inst.parts)))
        highest_offset = inst.parts[0].highestOffset
        notes_to_parse = inst.parts[0].recurse()
    except: # file has notes in a flat structure
        notes_to_parse = midi.flat.notes
        highest_offset = midi.flat.highestOffset

    print("Highest offset: %f" % highest_offset)

    for element in notes_to_parse:
        #if isinstance(element, note.Note) or isinstance(element, chord.Chord) or isinstance(element, note.Rest):
        #    note_and_offset = str(element) + str(element.offset)
        #   notes.append(str(note_and_offset))

        if isinstance(element, note.Note):
            if offset:
                current_offset = element.offset
                if offset_normalization:
                    current_offset = element.offset / highest_offset
                    current_offset = round(current_offset, offset_rounding)
                pitches.append(str(element.pitch) + "_" + str(current_offset))
            else:
                pitches.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            if offset:
                current_offset = element.offset
                if offset_normalization:
                    current_offset = element.offset / highest_offset
                    current_offset = round(current_offset, offset_rounding)
                pitches.append('.'.join(str(n) for n in element.normalOrder))
                pitches.append("_" + str(current_offset))
            else:
                pitches.append('.'.join(str(n) for n in element.normalOrder))
        elif isinstance(element, note.Rest) and rest == 1:
            if offset:
                current_offset = element.offset
                if offset_normalization:
                    current_offset = element.offset / highest_offset
                    current_offset = round(current_offset, offset_rounding)
                pitches.append("rest_" + str(current_offset))
            else:
                pitches.append("rest")

#for note in notes:
#    print(note)

pitchnames = sorted(set(item for item in pitches))
pitch_to_int = dict((pitches, number) for number, pitches in enumerate(pitchnames))
print("Dictionary size: %f" % len(pitch_to_int))
