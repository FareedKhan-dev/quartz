"""Errand: 1,200 device errands, 180 that must be refused, built from a seed.

No public suite has a wearable or a robot in it, and published benchmarks report
one accuracy number that conflates picking the tool with filling it in. So we
wrote our own, and the first thing to say about it is that we wrote the test we
then passed: its difficulty is our choice, which is why the comparisons against
other models are the trustworthy half of the result.

Four properties are built in on purpose, because a suite without them flatters
the model:

- **Held-out tools.** A fifth of the errands declare a tool the training data
  never contained, so the score is not a memory test.
- **Near-miss twins.** Every errand carries a twin differing in exactly one
  argument, with a *controlled* surface overlap against the original. A model
  that pattern-matches the phrasing gets one of the pair right and the other
  wrong.
- **Controlled distractors.** The five schemas rendered each turn are the target
  plus two tools chosen for name overlap with it and two chosen for none, so
  selection cannot be won by string similarity alone.
- **Off-topic input.** 180 requests no declared schema can serve. Refusal is a
  behaviour we train and therefore a behaviour we measure; with none of these in
  the training data the model calls a tool for 96.2 percent of nonsense.

Everything here is deterministic in `seed`: the same seed builds the same 1,380
records on any machine, with no network and no model. numpy is not even needed
-- this is `random` and `itertools`, so the suite can be built anywhere the
package imports.
"""
from __future__ import annotations

import functools
import itertools
import json
import random
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: The four device families and how many errands each contributes. The split is
#: roughly what a household of devices declares, not an equal partition: a phone
#: has the most surface, a robot the least.
FAMILIES: dict[str, int] = {"phone": 420, "wearable": 240, "smart home": 330,
                            "robot": 210}

#: Requests no declared schema can serve. Their gold answer is the empty list.
OFF_TOPIC = 180

#: Errands declaring a tool unseen in training, apportioned across the families
#: by largest remainder. 241 of 1,200 is 20.1 percent.
HELD_OUT = 241

#: Errands whose answer is two calls, as a share of each family.
MULTI_CALL_SHARE = 0.10

#: What Dowser renders into the window each turn: the target and four others.
TOOLS_PER_TURN = 5

#: Of those four, how many are chosen for surface overlap with the target.
NEAR_DISTRACTORS = 2

_WORD = re.compile(r"[a-z0-9]+")

#: The token overlap a twin is built to hit against its original, cycled across
#: the suite so the pairs span easy and hard discrimination rather than sitting
#: at one similarity the model could be tuned to.
OVERLAP_TARGETS: tuple[float, ...] = (0.60, 0.75, 0.90)


# --- the shape of a tool ---------------------------------------------------
@dataclass(frozen=True)
class Slot:
    """One argument, its JSON type, and the surfaces that may fill it.

    The surface written into the query *is* the argument value, so every
    argument in this suite is evidenced in its request. An errand whose gold
    answer contained a value the query never mentioned would be scoring
    invention, which is the one thing we are trying not to teach.
    """

    key: str
    type: str
    values: tuple[Any, ...]
    enum: bool = False
    minimum: int | None = None
    maximum: int | None = None

    def property_schema(self) -> dict[str, Any]:
        spec: dict[str, Any] = {"type": self.type}
        if self.enum:
            spec["enum"] = [str(v) for v in self.values]
        if self.minimum is not None:
            spec["minimum"] = self.minimum
        if self.maximum is not None:
            spec["maximum"] = self.maximum
        return spec


@dataclass(frozen=True)
class ToolSpec:
    """A declarable tool and the ways a person asks for it."""

    name: str
    description: str
    slots: tuple[Slot, ...]
    phrasings: tuple[str, ...]
    optional: tuple[str, ...] = ()
    held_out: bool = False
    family: str = ""

    def schema(self) -> dict[str, Any]:
        """The JSON schema, in the shape `agent.tools.build_schema` produces."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {s.key: s.property_schema() for s in self.slots},
                "required": [s.key for s in self.slots if s.key not in self.optional],
            },
        }

    def combinations(self) -> int:
        n = len(self.phrasings)
        for slot in self.slots:
            n *= len(slot.values)
        return n


@dataclass(frozen=True)
class PairSpec:
    """Two tools one sentence asks for, and how the second borrows from the first.

    `bind` maps an argument of the second call onto a slot of the first, which
    is what "look up Priya and text her" means: the recipient is the contact the
    first call searched for, and it is written once in the query.
    """

    first: str
    second: str
    phrasings: tuple[str, ...]
    bind: tuple[tuple[str, str], ...] = ()


@dataclass
class _Catalogue:
    """The tools of one family, split into what training saw and what it did not."""

    tools: tuple[ToolSpec, ...]
    pair: PairSpec
    seen: tuple[ToolSpec, ...] = field(default_factory=tuple)
    held: tuple[ToolSpec, ...] = field(default_factory=tuple)


# --- value pools -----------------------------------------------------------
CONTACTS = ("mum", "dad", "Priya", "Marcus", "Elena", "Yusuf", "Grandma",
            "Dr Okafor", "Sam", "Noor", "Theo", "Ines")
MESSAGES = ("running late", "on my way", "call me back", "dinner is at seven",
            "I landed", "leaving now", "see you at the gate", "bring the charger",
            "the meeting moved", "all done here")
PERCENTS = (5, 10, 20, 25, 30, 40, 50, 60, 75, 80, 90, 100)
MINUTES = (2, 3, 5, 8, 10, 12, 15, 20, 25, 30, 45, 90)
#: Pools cut to fit a narrower declared range. A value pool that steps outside
#: the schema it is declared under would make the suite's own gold answer
#: invalid, which `_check_catalogue` refuses at import.
SHORT_MINUTES = (2, 3, 5, 8, 10, 12, 15, 20, 25, 30, 45, 60)
LONG_MINUTES = (10, 15, 20, 25, 30, 40, 45, 60, 75, 90, 100, 120)
CLOCK = ("6:30", "7:00", "7:15", "8:00", "9:45", "11:20", "13:00", "16:30",
         "18:15", "21:00", "22:30", "5:45")
DAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
        "sunday", "tomorrow")
ALARM_LABELS = ("gym", "standup", "school run", "the flight", "medication",
                "laundry", "the interview", "stretching", "the bins", "yoga")
EVENTS = ("the dentist", "a standup", "lunch with Priya", "the school play",
          "a haircut", "the quarterly review", "a dog walk", "the car service")
TRACKS = ("the morning playlist", "something calm", "the album I saved",
          "jazz radio", "the podcast queue", "my running mix", "the study playlist",
          "the kids playlist", "the driving mix", "the sleep sounds",
          "the top of the charts", "last night's mix")
PHOTO_MODES = ("portrait", "panorama", "night", "macro", "burst", "selfie",
               "timelapse", "wide")
APPS = ("maps", "calendar", "camera", "notes", "podcasts", "weather", "banking",
        "messages", "gallery", "translate", "fitness", "wallet")
TRAVEL = ("car", "foot", "transit", "bike")
PLACES = ("the airport", "Ines' place", "the office", "the hardware store",
          "the clinic", "the school", "the ferry terminal", "the climbing gym",
          "the garden centre", "the train station")
ERRANDS = ("take the bins out", "call the dentist", "water the plants",
           "pay the invoice", "book the ferry", "charge the drone",
           "defrost the fish", "sign the form")
WHENS = ("tonight", "tomorrow morning", "on friday", "after lunch", "before bed",
         "at the weekend", "this evening", "on monday")
EXTENSIONS = (101, 118, 204, 219, 305, 330, 412, 447, 508, 522, 610, 634)

ACTIVITIES = ("run", "swim", "cycle", "row", "hike", "yoga", "walk", "elliptical")
MILLILITRES = (100, 150, 200, 250, 300, 330, 400, 500, 600, 750, 800, 1000)
STEPS = (2000, 3000, 4000, 5000, 6000, 7500, 8000, 9000, 10000, 12000, 15000, 20000)
HR_WINDOWS = ("now", "the last hour", "today", "this week", "the last ten minutes",
              "this morning", "yesterday", "the last workout")
KILOS = (52, 55, 58, 61, 64, 67, 70, 73, 76, 79, 82, 85)
ROUNDS = (3, 4, 5, 6, 8, 10, 12, 15)
NOTES = ("the knee felt fine", "slept badly", "new shoes today", "hills were hard",
         "cut the caffeine", "left calf tight", "wind on the back straight",
         "easy pace all through")
MOODS = ("calm", "tired", "focused", "restless", "cheerful", "flat", "anxious",
         "steady")

ROOMS = ("kitchen", "living room", "bedroom", "bathroom", "hallway", "garage",
         "office", "nursery", "patio", "basement", "attic", "dining room")
DOORS = ("front door", "back door", "garage door", "patio door", "side gate",
         "shed", "office door", "porch")
SCENES = ("movie night", "good morning", "away", "dinner", "reading", "party",
          "bedtime", "cleaning", "focus", "sunset", "guests", "workout")
CELSIUS = (16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27)
FAN_SPEEDS = ("off", "low", "medium", "high")
ALARM_MODES = ("home", "away", "night", "holiday", "guest", "pets", "perimeter",
               "garage only")
GARAGE_DOORS = ("main garage", "side garage", "workshop door", "carport gate",
                "roller shutter", "bay one", "bay two", "rear roller")
ZONES = ("the front beds", "the vegetable patch", "the greenhouse", "the herb boxes",
         "the back lawn", "the hanging baskets", "the fruit cage", "the side border")
HUMIDITY = (30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 78, 80)

SPOTS = ("the kitchen counter", "the charging dock", "the front door",
         "the packing bench", "the loading bay", "the sorting table",
         "the inspection cell", "the tool wall", "the paint booth", "the pallet stack")
OBJECTS = ("the red mug", "the blue crate", "the cardboard box", "the socket wrench",
           "the sample tray", "the cable spool", "the safety cone", "the label roll",
           "the empty jar", "the steel bracket")
DEGREES = (15, 30, 45, 60, 90, 120, 135, 180, 225, 270, 315, 360)
AREAS = ("the loading bay", "the north aisle", "the packing bench", "the cold store",
         "the yard", "the mezzanine", "the paint booth", "the dispatch lane")
WIDTHS = (10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120)
SUBSYSTEMS = ("the battery", "the gripper", "the lidar", "the wheels", "the arm",
              "the camera", "the bumper", "the wrist")
PATHS = ("the perimeter loop", "the aisle sweep", "the dock run", "the inspection lap",
         "the yard circuit", "the shortcut to dispatch", "the night patrol",
         "the delivery route")
JOINTS = ("the shoulder", "the elbow", "the wrist", "the base", "the gripper",
          "the forearm", "the pitch axis", "the roll axis")
PALLETS = ("pallet A", "pallet B", "pallet C", "pallet D", "pallet E", "pallet F",
           "pallet G", "pallet H")
HEIGHTS = (10, 15, 20, 25, 30, 40, 50, 60, 75, 90, 100, 120)


def _s(key: str, type_: str, values: Sequence[Any], **extra: Any) -> Slot:
    return Slot(key=key, type=type_, values=tuple(values), **extra)


def _tool(name: str, description: str, slots: Sequence[Slot],
          phrasings: Sequence[str], *, optional: Sequence[str] = (),
          held_out: bool = False) -> ToolSpec:
    return ToolSpec(name=name, description=description, slots=tuple(slots),
                    phrasings=tuple(phrasings), optional=tuple(optional),
                    held_out=held_out)


# --- the four catalogues ---------------------------------------------------
_PHONE = (
    _tool("send_message", "Send a text message to a contact.",
          [_s("recipient", "string", CONTACTS), _s("body", "string", MESSAGES)],
          ["text {recipient} saying {body}",
           "send {recipient} a message that says {body}",
           "let {recipient} know {body}",
           "message {recipient}: {body}"]),
    _tool("search_contact", "Find a person in the address book.",
          [_s("query", "string", CONTACTS)],
          ["look up {query} in my contacts",
           "find {query} in the address book",
           "search my contacts for {query}",
           "who is {query} in my contacts"]),
    _tool("call_contact", "Place a voice call to a contact.",
          [_s("name", "string", CONTACTS)],
          ["call {name}", "ring {name} for me", "give {name} a call",
           "dial {name} please"]),
    _tool("set_alarm", "Set an alarm for a time of day.",
          [_s("time", "string", CLOCK), _s("label", "string", ALARM_LABELS)],
          ["set an alarm for {time} labelled {label}",
           "wake me at {time} for {label}",
           "alarm at {time}, call it {label}",
           "put an alarm on at {time} for {label}"], optional=["label"]),
    _tool("start_timer", "Start a countdown timer.",
          [_s("minutes", "integer", MINUTES, minimum=1, maximum=180)],
          ["start a timer for {minutes} minutes",
           "set a {minutes} minute timer",
           "time {minutes} minutes for me",
           "count down {minutes} minutes"]),
    _tool("create_event", "Add an event to the calendar.",
          [_s("title", "string", EVENTS), _s("day", "string", DAYS),
           _s("hour", "string", CLOCK)],
          ["put {title} in my calendar on {day} at {hour}",
           "schedule {title} for {day} at {hour}",
           "add {title} to {day} at {hour}",
           "book {title} on {day} at {hour}"]),
    _tool("play_music", "Play a track, album or station.",
          [_s("track", "string", TRACKS)],
          ["play {track}", "put {track} on", "start playing {track}",
           "I want to hear {track}"]),
    _tool("set_volume", "Set the media volume.",
          [_s("level", "integer", PERCENTS, minimum=0, maximum=100)],
          ["set the volume to {level} percent",
           "turn the volume to {level} percent",
           "volume {level} percent please",
           "put the volume at {level} percent"]),
    _tool("take_photo", "Take a photograph in a camera mode.",
          [_s("mode", "string", PHOTO_MODES, enum=True)],
          ["take a {mode} photo", "shoot a {mode} picture",
           "camera, {mode} mode, take one", "grab a {mode} shot"]),
    _tool("open_app", "Open an installed application.",
          [_s("app", "string", APPS)],
          ["open {app}", "launch {app}", "bring up {app}", "get {app} on screen"]),
    _tool("mute_phone", "Silence the phone for a number of minutes.",
          [_s("minutes", "integer", MINUTES, minimum=1, maximum=180)],
          ["mute my phone for {minutes} minutes",
           "silence everything for {minutes} minutes",
           "no sound for the next {minutes} minutes",
           "quiet mode for {minutes} minutes"]),
    _tool("navigate_to", "Start turn-by-turn navigation.",
          [_s("destination", "string", PLACES),
           _s("mode", "string", TRAVEL, enum=True)],
          ["navigate to {destination} by {mode}",
           "take me to {destination} by {mode}",
           "directions to {destination} by {mode}",
           "route me to {destination} by {mode}"]),
    _tool("add_reminder", "Add a reminder for later.",
          [_s("text", "string", ERRANDS), _s("when", "string", WHENS)],
          ["remind me to {text} {when}",
           "set a reminder to {text} {when}",
           "don't let me forget to {text} {when}",
           "nudge me to {text} {when}"]),
    _tool("share_location", "Share live location with a contact for a while.",
          [_s("recipient", "string", CONTACTS),
           _s("duration_minutes", "integer", MINUTES, minimum=1, maximum=180)],
          ["share my location with {recipient} for {duration_minutes} minutes",
           "let {recipient} see where I am for {duration_minutes} minutes",
           "send {recipient} my live location for {duration_minutes} minutes",
           "location sharing with {recipient}, {duration_minutes} minutes"],
          held_out=True),
    _tool("read_notifications", "Read out the notifications of one app.",
          [_s("app", "string", APPS)],
          ["read me the notifications from {app}",
           "what has {app} been telling me",
           "catch me up on {app}",
           "read out anything new in {app}"], held_out=True),
    _tool("dial_extension", "Dial an internal extension number.",
          [_s("extension", "integer", EXTENSIONS, minimum=100, maximum=999)],
          ["dial extension {extension}", "put me through to extension {extension}",
           "call extension {extension}", "connect me to extension {extension}"],
          held_out=True),
)

_WEARABLE = (
    _tool("start_workout", "Begin a timed workout of one activity.",
          [_s("activity", "string", ACTIVITIES, enum=True),
           _s("minutes", "integer", MINUTES, minimum=1, maximum=240)],
          ["start a {minutes} minute {activity} workout",
           "begin a {activity} session for {minutes} minutes",
           "track a {activity} for {minutes} minutes",
           "log a {minutes} minute {activity}"]),
    _tool("log_water", "Record water drunk, in millilitres.",
          [_s("millilitres", "integer", MILLILITRES, minimum=10, maximum=2000)],
          ["log {millilitres} millilitres of water",
           "I drank {millilitres} millilitres",
           "add {millilitres} millilitres of water",
           "record {millilitres} millilitres of water"]),
    _tool("set_step_goal", "Set the daily step target.",
          [_s("steps", "integer", STEPS, minimum=500, maximum=50000)],
          ["set my step goal to {steps}",
           "make the daily target {steps} steps",
           "step goal {steps} from now on",
           "aim for {steps} steps a day"]),
    _tool("start_breathing", "Start a guided breathing exercise.",
          [_s("minutes", "integer", SHORT_MINUTES, minimum=1, maximum=60)],
          ["start a {minutes} minute breathing exercise",
           "breathe with me for {minutes} minutes",
           "guided breathing, {minutes} minutes",
           "calm me down for {minutes} minutes"]),
    _tool("heart_rate_summary", "Summarise heart rate over a window.",
          [_s("window", "string", HR_WINDOWS)],
          ["show me my heart rate for {window}",
           "what was my heart rate {window}",
           "heart rate summary for {window}",
           "how has my pulse been {window}"]),
    _tool("set_bedtime", "Set the bedtime reminder.",
          [_s("time", "string", CLOCK)],
          ["set my bedtime to {time}", "remind me to sleep at {time}",
           "bedtime at {time} please", "wind down at {time}"]),
    _tool("log_weight", "Record a weight in kilograms.",
          [_s("kilograms", "integer", KILOS, minimum=20, maximum=250)],
          ["log my weight as {kilograms} kilograms",
           "I weigh {kilograms} kilograms today",
           "record {kilograms} kilograms",
           "put {kilograms} kilograms in the log"]),
    _tool("start_interval", "Start an interval session.",
          [_s("rounds", "integer", ROUNDS, minimum=1, maximum=30),
           _s("minutes", "integer", SHORT_MINUTES, minimum=1, maximum=60)],
          ["start {rounds} intervals of {minutes} minutes",
           "{rounds} rounds, {minutes} minutes each",
           "interval session, {rounds} rounds of {minutes} minutes",
           "run {rounds} intervals at {minutes} minutes"]),
    _tool("set_stand_reminder", "Remind me to stand at an interval.",
          [_s("minutes", "integer", LONG_MINUTES, minimum=5, maximum=120)],
          ["remind me to stand every {minutes} minutes",
           "stand reminder every {minutes} minutes",
           "nudge me to get up every {minutes} minutes",
           "buzz me to stand each {minutes} minutes"]),
    _tool("record_note", "Save a short training note.",
          [_s("text", "string", NOTES)],
          ["note that {text}", "save a note: {text}",
           "log a note saying {text}", "write down that {text}"]),
    _tool("log_mood", "Record how the wearer feels.",
          [_s("mood", "string", MOODS, enum=True)],
          ["log my mood as {mood}", "I am feeling {mood}",
           "record that I feel {mood}", "put {mood} in the mood log"],
          held_out=True),
    _tool("start_nap", "Start a nap timer.",
          [_s("minutes", "integer", LONG_MINUTES, minimum=5, maximum=120)],
          ["start a {minutes} minute nap", "nap timer for {minutes} minutes",
           "wake me after a {minutes} minute nap",
           "let me sleep {minutes} minutes"], held_out=True),
)

_HOME = (
    _tool("set_lights", "Set the brightness of the lights in a room.",
          [_s("room", "string", ROOMS, enum=True),
           _s("brightness", "integer", PERCENTS, minimum=0, maximum=100)],
          ["dim the {room} to {brightness} percent",
           "set the {room} lights to {brightness} percent",
           "put the {room} at {brightness} percent",
           "{room} lights {brightness} percent please"]),
    _tool("set_thermostat", "Set a room's target temperature in celsius.",
          [_s("room", "string", ROOMS, enum=True),
           _s("celsius", "integer", CELSIUS, minimum=5, maximum=30)],
          ["set the {room} to {celsius} degrees",
           "make the {room} {celsius} degrees",
           "thermostat in the {room} to {celsius} degrees",
           "heat the {room} to {celsius} degrees"]),
    _tool("lock_door", "Lock one door or gate.",
          [_s("door", "string", DOORS, enum=True)],
          ["lock the {door}", "make sure the {door} is locked",
           "secure the {door}", "put the lock on the {door}"]),
    _tool("play_speaker", "Play audio on a room's speaker.",
          [_s("room", "string", ROOMS, enum=True), _s("track", "string", TRACKS)],
          ["play {track} in the {room}",
           "put {track} on the {room} speaker",
           "start {track} in the {room}",
           "{room} speaker, play {track}"]),
    _tool("run_scene", "Run a saved scene.",
          [_s("scene", "string", SCENES, enum=True)],
          ["run {scene}", "set the house to {scene}", "start the {scene} scene",
           "switch to {scene}"]),
    _tool("set_blinds", "Set how far a room's blinds are open.",
          [_s("room", "string", ROOMS, enum=True),
           _s("percent", "integer", PERCENTS, minimum=0, maximum=100)],
          ["open the {room} blinds to {percent} percent",
           "set the {room} blinds at {percent} percent",
           "{room} blinds {percent} percent",
           "close the {room} blinds to {percent} percent"]),
    _tool("start_vacuum", "Send the vacuum to clean a room.",
          [_s("room", "string", ROOMS, enum=True)],
          ["vacuum the {room}", "clean the {room} floor",
           "send the vacuum to the {room}", "hoover the {room}"]),
    _tool("arm_alarm", "Arm the alarm system in a mode.",
          [_s("mode", "string", ALARM_MODES, enum=True)],
          ["arm the alarm in {mode} mode", "set the alarm to {mode}",
           "put the alarm on {mode}", "alarm {mode} mode please"]),
    _tool("set_fan", "Set a room's fan speed.",
          [_s("room", "string", ROOMS, enum=True),
           _s("speed", "string", FAN_SPEEDS, enum=True)],
          ["set the {room} fan to {speed}", "{room} fan {speed} please",
           "turn the fan in the {room} to {speed}",
           "put the {room} fan on {speed}"]),
    _tool("boil_kettle", "Boil a volume of water.",
          [_s("millilitres", "integer", MILLILITRES, minimum=100, maximum=2000)],
          ["boil {millilitres} millilitres",
           "put {millilitres} millilitres on to boil",
           "kettle on, {millilitres} millilitres",
           "heat {millilitres} millilitres of water"]),
    _tool("water_plants", "Water a garden zone for a number of minutes.",
          [_s("zone", "string", ZONES),
           _s("minutes", "integer", SHORT_MINUTES, minimum=1, maximum=60)],
          ["water {zone} for {minutes} minutes",
           "run the irrigation on {zone} for {minutes} minutes",
           "give {zone} {minutes} minutes of water",
           "sprinklers on {zone}, {minutes} minutes"]),
    _tool("set_humidifier", "Set a room humidifier's target.",
          [_s("room", "string", ROOMS, enum=True),
           _s("percent", "integer", HUMIDITY, minimum=20, maximum=80)],
          ["set the {room} humidifier to {percent} percent",
           "humidity in the {room} at {percent} percent",
           "{room} humidifier {percent} percent",
           "keep the {room} at {percent} percent humidity"], held_out=True),
    _tool("open_garage", "Open one of the garage doors.",
          [_s("door", "string", GARAGE_DOORS, enum=True)],
          ["open the {door}", "let the {door} up", "raise the {door}",
           "unlock and open the {door}"], held_out=True),
    _tool("set_thermostat_schedule_weekday",
          "Set a weekday's scheduled temperature.",
          [_s("day", "string", DAYS, enum=True),
           _s("celsius", "integer", CELSIUS, minimum=5, maximum=30)],
          ["schedule {celsius} degrees for {day}",
           "on {day} the heating should be {celsius} degrees",
           "set the {day} schedule to {celsius} degrees",
           "every {day}, {celsius} degrees"], held_out=True),
)

_ROBOT = (
    _tool("move_to", "Drive the base to a named place.",
          [_s("destination", "string", SPOTS)],
          ["go to {destination}", "drive to {destination}",
           "move over to {destination}", "head to {destination}"]),
    _tool("pick_object", "Pick an object up from a place.",
          [_s("object", "string", OBJECTS), _s("location", "string", SPOTS)],
          ["pick up {object} from {location}",
           "grab {object} off {location}",
           "collect {object} at {location}",
           "take {object} from {location}"]),
    _tool("place_object", "Put a held object down at a place.",
          [_s("object", "string", OBJECTS), _s("destination", "string", SPOTS)],
          ["put {object} down at {destination}",
           "place {object} on {destination}",
           "set {object} down at {destination}",
           "leave {object} at {destination}"]),
    _tool("rotate_base", "Rotate the base by a number of degrees.",
          [_s("degrees", "integer", DEGREES, minimum=0, maximum=360)],
          ["rotate {degrees} degrees", "turn the base {degrees} degrees",
           "spin round {degrees} degrees", "yaw by {degrees} degrees"]),
    _tool("set_speed", "Set the drive speed as a percentage.",
          [_s("percent", "integer", PERCENTS, minimum=0, maximum=100)],
          ["set the speed to {percent} percent",
           "drive at {percent} percent",
           "slow to {percent} percent", "speed {percent} percent please"]),
    _tool("scan_area", "Scan an area and report what is there.",
          [_s("area", "string", AREAS)],
          ["scan {area}", "survey {area}", "have a look at {area}",
           "map {area} for me"]),
    _tool("open_gripper", "Open the gripper to a width in millimetres.",
          [_s("width_mm", "integer", WIDTHS, minimum=0, maximum=150)],
          ["open the gripper to {width_mm} millimetres",
           "gripper to {width_mm} millimetres",
           "set the gripper width to {width_mm} millimetres",
           "widen the gripper to {width_mm} millimetres"]),
    _tool("dock_charger", "Return to a charging dock.",
          [_s("dock", "string", SPOTS)],
          ["go and charge at {dock}", "dock at {dock}",
           "return to {dock} to charge", "plug yourself in at {dock}"]),
    _tool("report_status", "Report the state of one subsystem.",
          [_s("subsystem", "string", SUBSYSTEMS)],
          ["how is {subsystem}", "report on {subsystem}",
           "status of {subsystem} please", "check {subsystem}"]),
    _tool("follow_path", "Follow a saved path.",
          [_s("path", "string", PATHS)],
          ["follow {path}", "run {path}", "take {path}", "do {path} now"]),
    _tool("calibrate_arm", "Calibrate one joint of the arm.",
          [_s("joint", "string", JOINTS)],
          ["calibrate {joint}", "run a calibration on {joint}",
           "recalibrate {joint} please", "zero {joint}"], held_out=True),
    _tool("lift_pallet", "Lift a pallet to a height in centimetres.",
          [_s("pallet", "string", PALLETS),
           _s("height_cm", "integer", HEIGHTS, minimum=0, maximum=200)],
          ["lift {pallet} to {height_cm} centimetres",
           "raise {pallet} {height_cm} centimetres",
           "take {pallet} up to {height_cm} centimetres",
           "hoist {pallet} to {height_cm} centimetres"], held_out=True),
)

_PAIRS = {
    "phone": PairSpec(
        "search_contact", "send_message",
        ("look up {query} in my contacts and text them {body}",
         "find {query} in the address book then send: {body}",
         "search for {query} and message them {body}"),
        bind=(("recipient", "query"),)),
    "wearable": PairSpec(
        "heart_rate_summary", "start_workout",
        ("show me my heart rate for {window} then start a {minutes} minute {activity}"
         " workout",
         "heart rate for {window}, after that a {minutes} minute {activity}",
         "summarise my pulse {window} and begin a {activity} for {minutes} minutes")),
    "smart home": PairSpec(
        "run_scene", "lock_door",
        ("run {scene} and lock the {door}",
         "set the house to {scene}, then secure the {door}",
         "{scene} scene please, and lock the {door}")),
    "robot": PairSpec(
        "pick_object", "move_to",
        ("pick up {object} from {location} and go to {destination}",
         "grab {object} at {location} then drive to {destination}",
         "collect {object} from {location}, then head to {destination}")),
}

_OFF_TOPIC_TEMPLATES = (
    "what do you think about {topic}",
    "tell me your honest opinion on {topic}",
    "explain {topic} to me like I am five",
    "write me a short poem about {topic}",
    "why does everyone keep talking about {topic}",
    "how would you summarise {topic}",
    "is {topic} worth caring about",
    "give me three facts about {topic}",
    "what is the history of {topic}",
    "should I be worried about {topic}",
    "who decides what happens with {topic}",
    "settle an argument for me about {topic}",
)
_OFF_TOPIC_SUBJECTS = (
    "the situation in the Balkans", "the offside rule", "quantum computing",
    "the housing market", "sourdough starters", "the price of coffee",
    "medieval tapestries", "deep sea mining", "the electoral college",
    "tax policy", "string theory", "the anchovy fishery", "modern poetry",
    "the metric system", "volcanic winters", "the gig economy", "chess openings",
    "the Voyager probes", "antibiotic resistance", "urban foxes",
)

CATALOGUES: dict[str, _Catalogue] = {}
for _family, _tools in (("phone", _PHONE), ("wearable", _WEARABLE),
                        ("smart home", _HOME), ("robot", _ROBOT)):
    _tools = tuple(ToolSpec(**{**t.__dict__, "family": _family}) for t in _tools)
    CATALOGUES[_family] = _Catalogue(
        tools=_tools, pair=_PAIRS[_family],
        seen=tuple(t for t in _tools if not t.held_out),
        held=tuple(t for t in _tools if t.held_out))

#: Every tool in the suite, by name. The declared set of a turn is drawn from
#: here, so a distractor may come from another family: a real device does not
#: sort its catalogue by which box it lives in.
ALL_TOOLS: dict[str, ToolSpec] = {
    t.name: t for cat in CATALOGUES.values() for t in cat.tools}


def _check_catalogue() -> None:
    """Refuse to load a catalogue whose own gold answers would be invalid.

    Three ways to get this wrong, all of them silent: a phrasing that never
    mentions a slot, so the gold answer holds a value the query does not
    evidence; a value pool that steps outside the range declared beside it, so
    `schema_valid` can never reach 100 percent even with a perfect decoder; and
    a pair whose two tools collide on an argument key, so one call overwrites
    the other's value. Checking at import costs microseconds and turns each of
    them into a line number.
    """
    for name, spec in ALL_TOOLS.items():
        for slot in spec.slots:
            for phrasing in spec.phrasings:
                if "{" + slot.key + "}" not in phrasing:
                    raise ValueError(
                        f"{name}: phrasing {phrasing!r} never mentions {slot.key!r}, "
                        f"so its gold value would not be evidenced in the query")
            for value in slot.values:
                if slot.minimum is not None and value < slot.minimum:
                    raise ValueError(f"{name}.{slot.key}: {value} is below its "
                                     f"declared minimum {slot.minimum}")
                if slot.maximum is not None and value > slot.maximum:
                    raise ValueError(f"{name}.{slot.key}: {value} is above its "
                                     f"declared maximum {slot.maximum}")
                if slot.enum and str(value) != value:
                    raise ValueError(f"{name}.{slot.key}: an enum is declared as "
                                     f"strings, so {value!r} could never match")
    for family, catalogue in CATALOGUES.items():
        pair = catalogue.pair
        first, second = ALL_TOOLS[pair.first], ALL_TOOLS[pair.second]
        bound = dict(pair.bind)
        shared = ({s.key for s in first.slots} & {s.key for s in second.slots}
                  ) - set(bound)
        if shared:
            raise ValueError(
                f"{family}: {pair.first} and {pair.second} share the arguments "
                f"{sorted(shared)} without binding them, so one call would take "
                f"the other's value")


_check_catalogue()


# --- surface measures ------------------------------------------------------
def _words(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def surface_overlap(a: str, b: str) -> float:
    """Jaccard overlap of two requests, over lowercase word tokens.

    This is the number the twins are built against. A pair at 0.9 differs by one
    short word and is the hard case; a pair at 0.6 differs by a longer span and
    is the easy one.
    """
    wa, wb = _words(a), _words(b)
    union = wa | wb
    return len(wa & wb) / len(union) if union else 1.0


@functools.cache
def _tokens(spec: ToolSpec) -> frozenset[str]:
    """The surface of a tool: its name and its argument keys, split on `_`.

    Argument keys count because they are surface the model sees. `lock_door` and
    `open_garage` share no name token but both take a `door`, and a turn that
    declares them together is a harder selection than one that does not.
    """
    words = spec.name.split("_")
    for slot in spec.slots:
        words += slot.key.split("_")
    return frozenset(words)


def _tool_overlap(a: ToolSpec, b: ToolSpec) -> float:
    ta, tb = _tokens(a), _tokens(b)
    union = ta | tb
    return len(ta & tb) / len(union) if union else 0.0


# --- rendering one errand --------------------------------------------------
@functools.cache
def _neighbours(target: str, also: str = "") -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The tools nearest a target, and the ones sharing nothing with it.

    Cached: the ranking depends only on the catalogue, so computing it once per
    tool instead of once per errand is the difference between a suite that
    builds in a second and one that builds in ten.
    """
    spec = ALL_TOOLS[target]
    keep = {target, also} - {""}
    ranked = sorted((t for t in ALL_TOOLS.values() if t.name not in keep),
                    key=lambda t: (-_tool_overlap(t, spec), t.name))
    near = tuple(t.name for t in ranked[:NEAR_DISTRACTORS])
    far = tuple(t.name for t in ranked
                if t.name not in near and _tool_overlap(t, spec) == 0.0)
    return near, far


def _surface(value: Any) -> str:
    return f"{value:g}" if isinstance(value, float) else str(value)


def _apportion(total: int, weights: Mapping[str, int]) -> dict[str, int]:
    """Split `total` across `weights` by largest remainder, summing exactly."""
    scale = sum(weights.values())
    exact = {k: total * w / scale for k, w in weights.items()}
    out = {k: int(v) for k, v in exact.items()}
    order = sorted(weights, key=lambda k: exact[k] - out[k], reverse=True)
    for k in order[:total - sum(out.values())]:
        out[k] += 1
    return out


def _even(total: int, buckets: int) -> list[int]:
    """`total` split as evenly as possible, the remainder going to the first."""
    base, extra = divmod(total, max(1, buckets))
    return [base + (1 if i < extra else 0) for i in range(buckets)]


def _assignments(slots: Sequence[Slot], phrasings: Sequence[str],
                 rng: random.Random, count: int, what: str
                 ) -> list[tuple[str, dict[str, Any]]]:
    """`count` distinct (phrasing, slot values) pairs, shuffled.

    The whole product is enumerated and shuffled rather than sampled with
    rejection, so the suite is unique by construction and a catalogue too small
    to fill its share says so here instead of quietly repeating itself.
    """
    space = [(p, dict(zip([s.key for s in slots], combo, strict=True)))
             for p in phrasings
             for combo in itertools.product(*[s.values for s in slots])]
    if len(space) < count:
        raise ValueError(
            f"{what} can produce {len(space)} distinct errands, {count} asked for; "
            f"add a phrasing or widen a value pool")
    rng.shuffle(space)
    return space[:count]


def _reasoning(values: Mapping[str, Any]) -> str:
    """One line deriving every argument from the span it was copied out of."""
    return "; ".join(f"'{_surface(v)}' -> {k}" for k, v in values.items())


def _declare(target: ToolSpec, rng: random.Random, *,
             also: ToolSpec | None = None) -> tuple[list[dict[str, Any]], float]:
    """The five schemas rendered this turn, and how close the nearest one is.

    Two distractors are the closest tools by name overlap and the rest are drawn
    from the tools that share no name token at all. That is the "controlled" in
    controlled surface overlap: every turn has near misses in it by construction,
    so a model that picks a tool by string similarity to the request is measured
    on exactly the case where that heuristic breaks.
    """
    # A list, not a set: set iteration order over strings moves with the
    # interpreter's hash seed, and this suite has to be the same on every run.
    keep = [target.name] + ([also.name] if also is not None else [])
    near, far = _neighbours(target.name, keep[1] if len(keep) > 1 else "")
    room = max(0, TOOLS_PER_TURN - len(keep) - len(near))
    names = keep + list(near) + rng.sample(far, room)
    declared = [ALL_TOOLS[n] for n in names]
    rng.shuffle(declared)
    overlap = [_tool_overlap(t, target) for t in declared if t.name not in keep]
    return [t.schema() for t in declared], max(overlap or [0.0])


def _twin(phrasing: str, values: Mapping[str, Any], slots: Sequence[Slot],
          build: Any, target_overlap: float) -> dict[str, Any]:
    """The near miss: the same sentence with exactly one argument changed.

    Every single-slot substitution is scored by how close its overlap lands to
    the target band, and the closest wins. Changing "kitchen" to "bathroom" and
    changing it to "dining room" are different tests, and picking between them
    on purpose is what keeps the pairs from all being the same difficulty.
    """
    query = phrasing.format(**{k: _surface(v) for k, v in values.items()})
    best: tuple[float, str, str, dict[str, Any]] | None = None
    for slot in slots:
        for candidate in slot.values:
            if candidate == values[slot.key]:
                continue
            swapped = dict(values)
            swapped[slot.key] = candidate
            twin_query = phrasing.format(**{k: _surface(v)
                                            for k, v in swapped.items()})
            distance = abs(surface_overlap(query, twin_query) - target_overlap)
            if best is None or distance < best[0]:
                best = (distance, slot.key, twin_query, swapped)
    if best is None:                     # a tool with one value in one slot
        return {"query": query, "answers": build(values), "differs": "",
                "overlap": 1.0}
    _, key, twin_query, swapped = best
    return {"query": twin_query, "answers": build(swapped), "differs": key,
            "overlap": round(surface_overlap(query, twin_query), 4)}


def _single(tool: ToolSpec, rng: random.Random, count: int,
            overlaps: Iterator[float]) -> Iterator[dict[str, Any]]:
    """Errands answered by one call to `tool`."""
    def build(values: Mapping[str, Any]) -> list[dict[str, Any]]:
        return [{"name": tool.name, "arguments": dict(values)}]

    for phrasing, values in _assignments(tool.slots, tool.phrasings, rng, count,
                                         f"tool {tool.name!r}"):
        query = phrasing.format(**{k: _surface(v) for k, v in values.items()})
        tools, overlap = _declare(tool, rng)
        yield {
            "family": tool.family, "query": query, "tools": tools,
            "answers": build(values), "reasoning": _reasoning(values),
            "target": [tool.name], "held_out": tool.held_out, "off_topic": False,
            "distractor_overlap": round(overlap, 4),
            "twin": _twin(phrasing, values, tool.slots, build, next(overlaps)),
        }


def _multi(pair: PairSpec, rng: random.Random, count: int,
           overlaps: Iterator[float]) -> Iterator[dict[str, Any]]:
    """Errands answered by two calls, one sentence."""
    first, second = ALL_TOOLS[pair.first], ALL_TOOLS[pair.second]
    bound = dict(pair.bind)
    slots = list(first.slots) + [s for s in second.slots
                                 if s.key not in {x.key for x in first.slots}
                                 and s.key not in bound]

    def build(values: Mapping[str, Any]) -> list[dict[str, Any]]:
        a = {s.key: values[s.key] for s in first.slots}
        b = {s.key: (values[bound[s.key]] if s.key in bound else values[s.key])
             for s in second.slots}
        return [{"name": first.name, "arguments": a},
                {"name": second.name, "arguments": b}]

    for phrasing, values in _assignments(slots, pair.phrasings, rng, count,
                                         f"pair {pair.first}+{pair.second}"):
        query = phrasing.format(**{k: _surface(v) for k, v in values.items()})
        tools, overlap = _declare(first, rng, also=second)
        yield {
            "family": first.family, "query": query, "tools": tools,
            "answers": build(values), "reasoning": _reasoning(values),
            "target": [first.name, second.name], "held_out": False,
            "off_topic": False, "distractor_overlap": round(overlap, 4),
            "twin": _twin(phrasing, values, slots, build, next(overlaps)),
        }


def _off_topic(rng: random.Random, count: int) -> Iterator[dict[str, Any]]:
    """Requests nothing declared can serve, whose gold answer is `[]`.

    They still declare five tools. Refusal only means anything when there was
    something to call: an empty catalogue would test nothing at all.
    """
    space = [t.format(topic=s) for t in _OFF_TOPIC_TEMPLATES
             for s in _OFF_TOPIC_SUBJECTS]
    if len(space) < count:
        raise ValueError(f"only {len(space)} off-topic requests, {count} asked for")
    rng.shuffle(space)
    names = sorted(ALL_TOOLS)
    for query in space[:count]:
        declared = [ALL_TOOLS[n] for n in rng.sample(names, TOOLS_PER_TURN)]
        yield {
            "family": "off-topic", "query": query,
            "tools": [t.schema() for t in declared], "answers": [],
            "reasoning": "nothing declared here serves this request",
            "target": [], "held_out": False, "off_topic": True,
            "distractor_overlap": 0.0, "twin": None,
        }


# --- the suite -------------------------------------------------------------
def build_suite(seed: int = 0) -> list[dict[str, Any]]:
    """Build all 1,380 records: 1,200 errands and 180 that must be refused.

    Twenty percent of the errands declare a tool unseen in training, and each
    has a near-miss twin differing by one argument. Deterministic in `seed`, so
    a reported score names a suite anybody can rebuild byte for byte.

    Each record is::

        {"id", "family", "query", "tools", "answers", "reasoning", "target",
         "held_out", "off_topic", "distractor_overlap", "twin"}

    `tools` is the five schemas rendered into the window, `answers` is the gold
    call list, and `twin` is the near miss: `{"query", "answers", "differs",
    "overlap"}`, or None for an off-topic record.
    """
    rng = random.Random(seed)
    held_by_family = _apportion(HELD_OUT, FAMILIES)
    overlaps = itertools.cycle(OVERLAP_TARGETS)
    suite: list[dict[str, Any]] = []

    for family, total in FAMILIES.items():
        catalogue = CATALOGUES[family]
        held = held_by_family[family]
        multi = round(MULTI_CALL_SHARE * total)
        single = total - held - multi
        rows: list[dict[str, Any]] = []
        for tools, share in ((catalogue.held, held), (catalogue.seen, single)):
            for tool, count in zip(tools, _even(share, len(tools)), strict=True):
                rows.extend(_single(tool, rng, count, overlaps))
        rows.extend(_multi(catalogue.pair, rng, multi, overlaps))
        rng.shuffle(rows)
        slug = family.replace(" ", "-")
        for i, row in enumerate(rows, 1):
            suite.append({"id": f"{slug}-{i:04d}", **row})

    for i, row in enumerate(_off_topic(rng, OFF_TOPIC), 1):
        suite.append({"id": f"off-topic-{i:04d}", **row})
    return suite


def twin_errands(suite: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The near misses as scorable records of their own.

    Score the suite, score these, and the difference between the two is how much
    of the first score was the phrasing rather than the request.
    """
    out = []
    for row in suite:
        twin = row.get("twin")
        if not twin:
            continue
        out.append({**row, "id": f"{row['id']}-twin", "query": twin["query"],
                    "answers": twin["answers"],
                    "reasoning": _reasoning(
                        {k: v for call in twin["answers"]
                         for k, v in call["arguments"].items()}),
                    "twin": None, "twin_of": row["id"]})
    return out


def summary(suite: Sequence[Mapping[str, Any]]) -> str:
    """The four lines the post prints after building it."""
    errands = [r for r in suite if not r["off_topic"]]
    held = sum(1 for r in errands if r["held_out"])
    twins = [r["twin"]["overlap"] for r in errands if r.get("twin")]
    mean = sum(twins) / len(twins) if twins else 0.0
    return "\n".join([
        f"  {'suite':<10}{len(errands):,} errands + "
        f"{len(suite) - len(errands):,} off-topic",
        f"  {'held-out':<10}{held:,} errands ({held / max(1, len(errands)):.1%}) "
        f"declare a tool unseen in training",
        f"  {'twins':<10}{len(twins):,} near-miss pairs, mean surface overlap "
        f"{mean:.2f}",
        f"  {'catalogue':<10}{len(ALL_TOOLS):,} tools declared, "
        f"top {TOOLS_PER_TURN} rendered per turn",
    ])


def save_suite(suite: Sequence[Mapping[str, Any]], path: str) -> str:
    """Write the suite as JSON, which `quartz errand --suite` reads back."""
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps(list(suite), indent=2, ensure_ascii=False),
                    encoding="utf-8")
    return str(file)


def load_suite(path: str) -> list[dict[str, Any]]:
    """Read a saved suite back. The inverse of `save_suite`.

    A suite is worth saving when it has been scored: the seed reproduces the
    errands, but a file is what lets two runs be compared errand by errand
    without trusting that the generator has not moved underneath them.
    """
    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(
            f"no suite at {file}. Build one with `build_suite(seed=0)`, or pass "
            f"no --suite at all and it will be built from the seed.")
    return json.loads(file.read_text(encoding="utf-8"))
