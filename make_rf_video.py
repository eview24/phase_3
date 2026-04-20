"""
Random Forest Regression Model — Equation Animation
=====================================================
Animates the equations of the random forest model, step by step.

RENDER:
    manim -pql make_rf_video.py RandomForestModel
    manim -pqh make_rf_video.py RandomForestModel   # high quality
"""

from manim import *


class RandomForestModel(Scene):

    def construct(self):

        # ── Helpers ──────────────────────────────────────────────────────────
        def title_card(text, color=WHITE):
            t = Text(text, font_size=28, color=color)
            t.to_edge(UP, buff=0.3)
            return t

        def label(text, font_size=24, color=GREY_B):
            return Text(text, font_size=font_size, color=color)

        # ── STEP 0: Intro title ──────────────────────────────────────────────
        intro = Text("Random Forest Regression", font_size=40, color=WHITE)
        subtitle = Text(
            "How the model is built up, layer by layer.",
            font_size=24, color=GREY_B,
        )
        subtitle.next_to(intro, DOWN, buff=0.3)

        self.play(Write(intro), run_time=1.5)
        self.play(FadeIn(subtitle, shift=DOWN * 0.2))
        self.wait(2)
        self.play(FadeOut(intro), FadeOut(subtitle))

        # ── STEP 1: Single decision tree ─────────────────────────────────────
        hdr = title_card("Step 1 — A Single Decision Tree")
        self.play(FadeIn(hdr))

        eq_tree = MathTex(
            r"\hat{y}_i = T(\mathbf{x}_i)",
            font_size=64,
        )
        eq_tree.center()

        self.play(Write(eq_tree), run_time=2)

        note1 = label(
            "Each observation i is one club over one season.\n"
            "One tree recursively splits the feature space into regions.\n",          
            font_size=22,
        )
        note1.next_to(eq_tree, DOWN, buff=0.6)
        self.play(FadeIn(note1, shift=DOWN * 0.2))
        self.wait(3)
        self.play(FadeOut(note1))

        # ── STEP 2: Bootstrap aggregation (bagging) ──────────────────────────
        hdr2 = title_card("Step 2 — Bootstrap Aggregation (Bagging)")
        self.play(Transform(hdr, hdr2))

        self.play(
            eq_tree.animate.to_edge(UP, buff=1.0).shift(DOWN * 0.4),
            run_time=0.8,
        )

        eq_forest = MathTex(
            r"\hat{y}_i = \frac{1}{B} \sum_{b=1}^{B} T_b\!\left(\mathbf{x}_i;\; \mathcal{D}^{(b)}\right)",
            font_size=52,
        )
        eq_forest.set_color(BLUE)
        eq_forest.next_to(eq_tree, DOWN, buff=0.9)

        self.play(Write(eq_forest), run_time=2)

        note2 = label(
            "Each tree trains on a bootstrap sample of the data, averaging across trees cancels noise.",
            font_size=22,
        )
        note2.next_to(eq_forest, DOWN, buff=0.5)
        self.play(FadeIn(note2, shift=DOWN * 0.2))
        self.wait(2.5)
        self.play(FadeOut(note2))

        # ── STEP 3: Feature subsampling at each split ─────────────────────────
        hdr3 = title_card("Step 3 — Feature Subsampling at Each Split")
        self.play(Transform(hdr, hdr3))

        self.play(
            FadeOut(eq_tree),
            eq_forest.animate.to_edge(UP, buff=1.0).shift(DOWN * 0.4),
            run_time=0.8,
        )

        eq_mtry = MathTex(
            r"\mathcal{S} \subset \{1, \ldots, p\}",
            r"\qquad",
            r"|\mathcal{S}| = m_{\text{try}}",
            font_size=52,
        )
        eq_mtry.set_color(RED)
        eq_mtry.next_to(eq_forest, DOWN, buff=0.9)

        self.play(Write(eq_mtry), run_time=1.5)

        note3 = label(
            "At each node only m_try randomly chosen features are considered for splitting.\n"
            "This decorrelates the trees.",
            font_size=22,
        )
        note3.next_to(eq_mtry, DOWN, buff=0.5)
        self.play(FadeIn(note3, shift=DOWN * 0.2))
        self.wait(2.5)
        self.play(FadeOut(note3))

        # ── STEP 4: League-aware features ────────────────────────────────────
        hdr4 = title_card("Step 4 — League-Aware Features")
        self.play(Transform(hdr, hdr4))

        self.play(
            FadeOut(eq_forest),
            eq_mtry.animate.to_edge(UP, buff=1.0).shift(DOWN * 0.4),
            run_time=0.8,
        )

        eq_league = MathTex(
            r"\tilde{\mathbf{x}}_i = \bigl(\mathbf{x}_i,\; \mathrm{onehot}(g(i))\bigr)",
            font_size=52,
        )
        eq_league.set_color(GREEN)
        eq_league.next_to(eq_mtry, DOWN, buff=0.9)

        self.play(Write(eq_league), run_time=1.5)

        note4 = label(
            "One-hot league indicators let trees split on league.",
            font_size=22,
        )
        note4.next_to(eq_league, DOWN, buff=0.5)
        self.play(FadeIn(note4, shift=DOWN * 0.2))
        self.wait(2.5)
        self.play(FadeOut(note4))

        # ── STEP 5: OOB residuals and prediction intervals ───────────────────
        hdr5 = title_card("Step 5 — Out-of-Bag Residuals & Prediction Intervals")
        self.play(Transform(hdr, hdr5))

        self.play(
            FadeOut(eq_mtry),
            eq_league.animate.to_edge(UP, buff=1.0).shift(DOWN * 0.4),
            run_time=0.8,
        )

        eq_oob = MathTex(
            r"r_i^{\text{OOB}} = y_i - \frac{1}{|\mathcal{B}_i^{-}|} \sum_{b \notin \mathcal{B}_i} T_b(\tilde{\mathbf{x}}_i)",
            font_size=40,
        )
        eq_oob.set_color(YELLOW)
        eq_oob.next_to(eq_league, DOWN, buff=0.8)

        eq_interval = MathTex(
            r"\hat{y}_i \;\pm\; z_{\alpha/2}\, \hat{\sigma}_{\text{OOB}}",
            font_size=48,
        )
        eq_interval.set_color(YELLOW)
        eq_interval.next_to(eq_oob, DOWN, buff=0.6)

        self.play(Write(eq_oob), run_time=2)
        self.play(Write(eq_interval), run_time=1.5)

        note5 = label(
            "Residuals from trees that didn't see row i give a free, honest interval.",
            font_size=22,
        )
        note5.next_to(eq_interval, DOWN, buff=0.5)
        self.play(FadeIn(note5, shift=DOWN * 0.2))
        self.wait(2.5)
        self.play(FadeOut(note5), FadeOut(eq_league), FadeOut(eq_oob), FadeOut(eq_interval), FadeOut(hdr))

        # ── FINAL: Full Model Summary ─────────────────────────────────────────
        summary_title = Text("Full Model Summary", font_size=36, color=WHITE)
        summary_title.to_edge(UP, buff=0.3)
        self.play(FadeIn(summary_title))

        lines = [
            (r"\hat{y}_i = T(\mathbf{x}_i)", WHITE),
            (r"\hat{y}_i = \frac{1}{B} \sum_{b=1}^{B} T_b(\mathbf{x}_i;\; \mathcal{D}^{(b)})", BLUE),
            (r"\mathcal{S} \subset \{1,\ldots,p\}, \quad |\mathcal{S}| = m_{\text{try}}", RED),
            (r"\tilde{\mathbf{x}}_i = \bigl(\mathbf{x}_i,\; \mathrm{onehot}(g(i))\bigr)", GREEN),
            (r"r_i^{\text{OOB}} = y_i - \frac{1}{|\mathcal{B}_i^{-}|}\sum_{b \notin \mathcal{B}_i} T_b(\tilde{\mathbf{x}}_i)", YELLOW),
            (r"\hat{y}_i \pm z_{\alpha/2}\,\hat{\sigma}_{\text{OOB}}", YELLOW),
        ]

        summary = VGroup(*[
            MathTex(line, font_size=26).set_color(col)
            for line, col in lines
        ])
        summary.arrange(DOWN, aligned_edge=LEFT, buff=0.32)
        summary.center()

        self.play(
            LaggedStart(*[Write(line) for line in summary], lag_ratio=0.4),
            run_time=3,
        )

        box = SurroundingRectangle(summary, color=BLUE_B, buff=0.25, stroke_width=2)
        self.play(Create(box))
        self.wait(3)
