"""
Bayesian Hierarchical Regression Model — Equation Animation
============================================================
Animates the equations of a hierarchical Bayesian regression model,
transforming variables into one another as the model is explained.

RENDER:
    manim -pql bayesian_model_animation.py BayesianHierarchicalModel
    manim -pqh bayesian_model_animation.py BayesianHierarchicalModel   # high quality
"""

from manim import *


class BayesianHierarchicalModel(Scene):

    def construct(self):

        # ── Helpers ──────────────────────────────────────────────────────────
        def title_card(text, color=WHITE):
            t = Text(text, font_size=28, color=color)
            t.to_edge(UP, buff=0.3)
            return t

        def label(text, font_size=24, color=GREY_B):
            return Text(text, font_size=font_size, color=color)

        # ── STEP 0: Intro title ──────────────────────────────────────────────
        intro = Text(
            "Bayesian Hierarchical Regression",
            font_size=40, color=WHITE,
        )
        subtitle = Text(
            "How the model is built up, layer by layer.",
            font_size=24, color=GREY_B,
        )
        subtitle.next_to(intro, DOWN, buff=0.3)

        self.play(Write(intro), run_time=1.5)
        self.play(FadeIn(subtitle, shift=DOWN * 0.2))
        self.wait(2)
        self.play(FadeOut(intro), FadeOut(subtitle))

        # ── STEP 1: Base likelihood  y_i ~ N(mu_i, sigma) ───────────────────
        hdr = title_card("Step 1 — Standard Bayesian regression")
        self.play(FadeIn(hdr))

        eq_likelihood = MathTex(
            r"y_i \sim \mathcal{N}(\mu_i,\, \sigma)",
            font_size=64,
        )
        eq_likelihood.set_color(WHITE)

        note1 = label("Each observation i is one club over one season.")
        note1.next_to(eq_likelihood, DOWN, buff=0.6)

        self.play(Write(eq_likelihood), run_time=1.5)
        self.play(FadeIn(note1, shift=DOWN * 0.2))
        self.wait(2)

        # Highlight mu_i — it will be expanded next
        self.play(
            eq_likelihood[0][5:7].animate.set_color(BLUE),   # mu_i
            run_time=2.0,
        )
        self.wait(1)
        self.play(FadeOut(note1))

        # ── STEP 2: Expand mu_i into league-level linear predictor ───────────
        hdr2 = title_card("Step 2 — League-level mean (hierarchy introduced)")
        self.play(Transform(hdr, hdr2))

        eq_mu = MathTex(
            r"\mu_i = \alpha_{g(i)} + \sum_j \beta_{g(i),j}\, x_{ij}",
            font_size=56,
        )
        eq_mu.next_to(eq_likelihood, DOWN, buff=0.9)

        # Animate mu_i morphing into the expanded form
        mu_copy = eq_likelihood[0][4:7].copy()
        self.play(
            TransformMatchingShapes(mu_copy, eq_mu),
            run_time=0.9,
        )
        self.wait(0.5)

        note2 = label("g(i) = league of observation i   |   each league gets its own α and β.")
        note2.next_to(eq_mu, DOWN, buff=0.5)
        self.play(FadeIn(note2, shift=DOWN * 0.2))
        self.wait(2.5)

        # Highlight alpha and beta — they'll get priors next
        self.play(
            eq_mu[0][3:8].animate.set_color(GREEN),    # alpha_{g(i)}
            eq_mu[0][11:18].animate.set_color(PINK),   # beta_{g(i),j}
            run_time=0.9,
        )
        self.wait(1)

        self.play(FadeOut(note2))
        self.play(
            FadeOut(eq_likelihood),
            eq_mu.animate.to_edge(UP, buff=1.2).shift(DOWN * 0.6),
            run_time=1,
        )

        # ── STEP 3: Hierarchical priors on alpha and beta ────────────────────
        hdr3 = title_card("Step 3 — Hierarchical priors (partial pooling)")
        self.play(Transform(hdr, hdr3))

        eq_priors = MathTex(
            r"\alpha_{g} \sim \mathcal{N}(\mu_\alpha, \sigma_\alpha)",
            r"\quad",
            r"\beta_{g,j} \sim \mathcal{N}(\mu_{\beta,j}, \sigma_{\beta,j})",
            font_size=44,
        )
        eq_priors.next_to(eq_mu, DOWN, buff=0.7)
        eq_priors[0].set_color(GREEN)
        eq_priors[2].set_color(PINK)

        self.play(Write(eq_priors), run_time=2)

        note3 = label("Leagues share information through common hyperparameters.")
        note3.next_to(eq_priors, DOWN, buff=0.5)
        self.play(FadeIn(note3, shift=DOWN * 0.2))
        self.wait(2.5)

        # Highlight the hyperparameters mu_alpha, sigma_alpha etc.
        self.play(
            eq_priors[0][4:].animate.set_color(ORANGE),
            eq_priors[2][6:].animate.set_color(YELLOW),
            run_time=1.8,
        )
        self.wait(1)
        self.play(FadeOut(note3))

        # ── STEP 4: Hyperpriors on intercept hyperparameters ─────────────────
        self.play(
            FadeOut(eq_mu),
            eq_priors.animate.to_edge(UP, buff=1.0).shift(DOWN * 0.5),
            run_time=0.8,
        )

        hdr4 = title_card("Step 4 — Hyperpriors on intercept parameters")
        self.play(Transform(hdr, hdr4))

        eq_hyper_alpha = MathTex(
            r"\mu_\alpha \sim \mathcal{N}(0, 1)",
            r"\quad",
            r"\sigma_\alpha \sim \text{Half-Normal}(0,\, \tau_{\sigma_\alpha})",
            font_size=44,
        )
        eq_hyper_alpha.next_to(eq_priors, DOWN, buff=0.8)
        eq_hyper_alpha[0].set_color(ORANGE)
        eq_hyper_alpha[2].set_color(ORANGE)

        # Transform the orange mu_alpha / sigma_alpha in eq_priors into new eq
        hyper_copy = eq_priors[0][5:].copy()
        self.play(
            TransformMatchingShapes(hyper_copy, eq_hyper_alpha),
            run_time=1.8,
        )
        self.wait(2.5)

        # ── STEP 5: Hyperpriors on slope hyperparameters ─────────────────────
        self.play(
            FadeOut(eq_priors),
            eq_hyper_alpha.animate.to_edge(UP, buff=1.0).shift(DOWN * 0.5),
            run_time=0.8,
        )

        hdr5 = title_card("Step 5 — Hyperpriors on slope parameters")
        self.play(Transform(hdr, hdr5))

        eq_hyper_beta = MathTex(
            r"\mu_{\beta,j} \sim \mathcal{N}(\mu_{\beta,j}^{\mu},\, \mu_{\beta,j}^{\sigma})",
            r"\quad",
            r"\sigma_{\beta,j} \sim \text{Half-Normal}(0,\, \sigma_{\beta,j}^{\alpha})",
            font_size=38,
        )
        eq_hyper_beta.next_to(eq_hyper_alpha, DOWN, buff=0.7)
        eq_hyper_beta[0].set_color(YELLOW)
        eq_hyper_beta[2].set_color(YELLOW)

        self.play(Write(eq_hyper_beta), run_time=2)

        note5 = label("A further level of hierarchy governs the slopes.")
        note5.next_to(eq_hyper_beta, DOWN, buff=0.5)
        self.play(FadeIn(note5, shift=DOWN * 0.2))
        self.wait(1.5)

        self.play(FadeOut(note5))
      

        # ── FINAL: Show full model summary ────────────────────────────────────
        self.play(
            FadeOut(eq_hyper_alpha),
            FadeOut(eq_hyper_beta),
            FadeOut(hdr),
            run_time=0.8,
        )

        summary_title = Text("Full Model Summary", font_size=36, color=WHITE)
        summary_title.to_edge(UP, buff=0.3)
        self.play(FadeIn(summary_title))

        lines = [
            r"y_i \sim \mathcal{N}(\mu_i, \sigma)",
            r"\mu_i = \alpha_{g(i)} + \textstyle\sum_j \beta_{g(i),j}\, x_{ij}",
            r"\alpha_g \sim \mathcal{N}(\mu_\alpha, \sigma_\alpha)",
            r"\beta_{g,j} \sim \mathcal{N}(\mu_{\beta,j}, \sigma_{\beta,j})",
            r"\mu_\alpha \sim \mathcal{N}(0, 1) \quad \sigma_\alpha \sim \text{Half-Normal}(0,\, \tau_{\sigma_\alpha})",
            r"\mu_{\beta,j} \sim \mathcal{N}(\mu_{\beta,j}^{\mu},\, \mu_{\beta,j}^{\sigma}) \quad \sigma_{\beta,j} \sim \text{Half-Normal}(0,\, \sigma_{\beta,j}^{\alpha})",
        ]
        
        colours = [WHITE, YELLOW, GREEN, PINK, ORANGE, YELLOW]
        
        summary = VGroup(*[
            MathTex(line, font_size=26).set_color(col)
            for line, col in zip(lines, colours)
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
