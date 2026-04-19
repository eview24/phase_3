"""
Manim Equation Animation Template
===================================
Inspired by 3Blue1Brown's style.

SETUP:
    pip install manim

RENDER COMMANDS:
    manim -pql manim_equations_template.py SceneName   # low quality, quick preview
    manim -pqh manim_equations_template.py SceneName   # high quality
    manim -pqm manim_equations_template.py SceneName   # medium quality

    Flags:  -p = play after render  |  -q = quality  |  -s = save last frame
"""

from manim import *


# ─────────────────────────────────────────────────────────────────────────────
# 1. WRITE AN EQUATION (basic)
# ─────────────────────────────────────────────────────────────────────────────
class WriteEquation(Scene):
    """Writes a LaTeX equation onto the screen, letter by letter."""

    def construct(self):
        # MathTex renders LaTeX math.  Use Tex() for plain text.
        equation = MathTex(r"e^{i\pi} + 1 = 0", font_size=96)
        equation.set_color(WHITE)

        title = Text("Euler's Identity", font_size=36, color=BLUE)
        title.next_to(equation, UP, buff=0.5)

        self.play(Write(equation), run_time=2)
        self.play(FadeIn(title, shift=UP * 0.3))
        self.wait(2)


# ─────────────────────────────────────────────────────────────────────────────
# 2. TRANSFORM ONE EQUATION INTO ANOTHER
# ─────────────────────────────────────────────────────────────────────────────
class TransformEquation(Scene):
    """
    Shows algebraic steps by morphing one equation into the next.
    TransformMatchingTex tries to match identical substrings and
    animates them in-place — great for showing algebra.
    """

    def construct(self):
        steps = [
            r"x^2 - 5x + 6 = 0",
            r"(x - 2)(x - 3) = 0",
            r"x = 2 \quad \text{or} \quad x = 3",
        ]

        label = Text("Solving a quadratic", font_size=30, color=YELLOW)
        label.to_edge(UP)
        self.play(FadeIn(label))

        current = MathTex(steps[0], font_size=64)
        self.play(Write(current))
        self.wait(1)

        for next_step in steps[1:]:
            next_eq = MathTex(next_step, font_size=64)
            self.play(TransformMatchingShapes(current, next_eq), run_time=1.5)
            current = next_eq
            self.wait(1.2)

        self.wait(1)


# ─────────────────────────────────────────────────────────────────────────────
# 3. COLOUR-HIGHLIGHT PARTS OF AN EQUATION
# ─────────────────────────────────────────────────────────────────────────────
class ColourEquationParts(Scene):
    """
    Writes an equation, then highlights individual terms in different colours
    with matching labels — useful for explaining what each symbol means.
    """

    def construct(self):
        # Split the equation into individually addressable parts.
        # Each string in the list becomes a separate submobject you can colour.
        eq = MathTex(
            r"a", r"x^2", r"+", r"b", r"x", r"+", r"c", r"= 0",
            font_size=72,
        )
        self.play(Write(eq))
        self.wait(0.5)

        # Colour map: index → colour
        colours = {0: RED, 1: RED, 3: BLUE, 4: BLUE, 6: GREEN}
        animations = [eq[i].animate.set_color(c) for i, c in colours.items()]
        self.play(*animations)

        # Add labels below each coloured term
        label_a = Text("quadratic\nterm", font_size=22, color=RED).next_to(eq[1], DOWN, buff=0.7)
        label_b = Text("linear\nterm",    font_size=22, color=BLUE).next_to(eq[4], DOWN, buff=0.7)
        label_c = Text("constant",        font_size=22, color=GREEN).next_to(eq[6], DOWN, buff=0.7)

        self.play(
            FadeIn(label_a, shift=DOWN * 0.2),
            FadeIn(label_b, shift=DOWN * 0.2),
            FadeIn(label_c, shift=DOWN * 0.2),
        )
        self.wait(2)


# ─────────────────────────────────────────────────────────────────────────────
# 4. ANIMATE A GRAPH + EQUATION TOGETHER
# ─────────────────────────────────────────────────────────────────────────────
class GraphWithEquation(Scene):
    """Plots a function and shows its equation, with a moving dot tracer."""

    def construct(self):
        # --- Axes ---
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-1, 9, 1],
            x_length=7,
            y_length=5,
            axis_config={"color": GREY},
            tips=False,
        )
        axes.to_edge(LEFT, buff=1)
        labels = axes.get_axis_labels(x_label="x", y_label="y")

        # --- Curve ---
        graph = axes.plot(lambda x: x**2, color=YELLOW, stroke_width=3)

        # --- Equation label ---
        eq = MathTex(r"y = x^2", font_size=48, color=YELLOW)
        eq.to_corner(UR).shift(LEFT * 0.5)

        self.play(Create(axes), Write(labels))
        self.play(Create(graph), Write(eq), run_time=2)

        # --- Moving dot along curve ---
        dot = Dot(color=RED)
        dot.move_to(axes.c2p(-3, 9))

        x_tracker = ValueTracker(-3)
        dot.add_updater(
            lambda d: d.move_to(axes.c2p(x_tracker.get_value(), x_tracker.get_value() ** 2))
        )

        self.add(dot)
        self.play(x_tracker.animate.set_value(3), run_time=4, rate_func=linear)
        dot.clear_updaters()
        self.wait(1)


# ─────────────────────────────────────────────────────────────────────────────
# 5. STEP-BY-STEP PROOF (equations appearing line by line)
# ─────────────────────────────────────────────────────────────────────────────
class StepByStepProof(Scene):
    """
    Shows a multi-step derivation where each line appears sequentially,
    mimicking a chalkboard worked example.
    """

    def construct(self):
        steps = [
            (r"\text{Pythagorean theorem:}", WHITE),
            (r"a^2 + b^2 = c^2",            YELLOW),
            (r"c^2 = a^2 + b^2",            YELLOW),
            (r"c   = \sqrt{a^2 + b^2}",     GREEN),
        ]

        group = VGroup()
        for i, (tex, colour) in enumerate(steps):
            mob = MathTex(tex, font_size=48, color=colour) if not tex.startswith(r"\text") \
                  else MathTex(tex, font_size=36, color=colour)
            mob.to_edge(LEFT, buff=1.5)
            if i == 0:
                mob.to_edge(UP, buff=1.5).to_edge(LEFT, buff=1.5)
            else:
                mob.next_to(group[-1], DOWN, aligned_edge=LEFT, buff=0.4)
            group.add(mob)
            self.play(Write(mob), run_time=0.8 if i == 0 else 1.2)
            self.wait(0.6)

        # Highlight the final result
        box = SurroundingRectangle(group[-1], color=GREEN, buff=0.15, stroke_width=3)
        self.play(Create(box))
        self.wait(2)


# ─────────────────────────────────────────────────────────────────────────────
# 6. NUMBER LINE / VALUE TRACKER (animating a changing value)
# ─────────────────────────────────────────────────────────────────────────────
class ValueTrackerDemo(Scene):
    """
    Displays a value that changes smoothly, with the equation updating live.
    Useful for showing limits, derivatives, or parameter sweeps.
    """

    def construct(self):
        tracker = ValueTracker(0)

        # DecimalNumber updates automatically when linked to a tracker
        value_display = DecimalNumber(
            0, num_decimal_places=2, font_size=72, color=TEAL
        )
        value_display.add_updater(lambda d: d.set_value(tracker.get_value()))
        value_display.to_edge(UP)

        label = MathTex(r"n = ", font_size=72, color=WHITE)
        label.next_to(value_display, LEFT)

        # A bar that grows with the value
        bar = Rectangle(width=0, height=0.6, color=TEAL, fill_opacity=0.8)
        bar.to_edge(LEFT, buff=1).shift(DOWN * 0.5)
        bar.add_updater(
            lambda b: b.become(
                Rectangle(
                    width=max(0.01, tracker.get_value() / 10 * 6),
                    height=0.6,
                    color=TEAL,
                    fill_opacity=0.8,
                ).to_edge(LEFT, buff=1).shift(DOWN * 0.5)
            )
        )

        self.add(label, value_display, bar)
        self.play(tracker.animate.set_value(10), run_time=4, rate_func=smooth)
        self.wait(1)
        self.play(tracker.animate.set_value(3),  run_time=2, rate_func=smooth)
        self.wait(1)


# ─────────────────────────────────────────────────────────────────────────────
# QUICK REFERENCE
# ─────────────────────────────────────────────────────────────────────────────
# Key Manim objects:
#   MathTex(r"...")          – LaTeX math
#   Tex(r"...")              – LaTeX text
#   Text("...")              – plain text (no LaTeX)
#   MathTex("a","b","c")     – multi-part eq (each part addressable by index)
#   Axes(...)                – coordinate system
#   axes.plot(func)          – plot a function
#   NumberLine(...)          – a single axis
#   Dot / Circle / Arrow     – geometric primitives
#   ValueTracker(n)          – animatable scalar
#   DecimalNumber(n)         – on-screen number that can follow a tracker
#
# Key animations:
#   Write(mob)                      – draw like handwriting
#   Create(mob)                     – draw stroke-by-stroke
#   FadeIn / FadeOut                – opacity
#   Transform(a, b)                 – morph a into b (a is removed)
#   ReplacementTransform(a, b)      – same but replaces a with b
#   TransformMatchingShapes(a, b)   – smart morph matching similar parts
#   TransformMatchingTex(a, b)      – match by LaTeX substring
#   mob.animate.set_color(c)        – change colour smoothly
#   mob.animate.shift(direction)    – move
#   mob.animate.scale(factor)       – resize
#
# Useful colours:   RED BLUE GREEN YELLOW ORANGE PURPLE TEAL WHITE GREY
# Useful directions: UP DOWN LEFT RIGHT UR UL DR DL
# ─────────────────────────────────────────────────────────────────────────────
