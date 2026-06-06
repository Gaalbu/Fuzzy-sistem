from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from textwrap import fill

import numpy as np
import matplotlib.pyplot as plt


def trimf(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    y = np.zeros_like(x, dtype=float)
    left = (a < x) & (x < b)
    right = (b <= x) & (x < c)
    y[left] = (x[left] - a) / (b - a)
    y[right] = (c - x[right]) / (c - b)
    y[x == b] = 1.0
    return np.clip(y, 0.0, 1.0)


def trapmf(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    y = np.zeros_like(x, dtype=float)
    if b > a:
        rising = (a < x) & (x < b)
        y[rising] = (x[rising] - a) / (b - a)
    y[(b <= x) & (x <= c)] = 1.0
    if d > c:
        falling = (c < x) & (x < d)
        y[falling] = (d - x[falling]) / (d - c)
    return np.clip(y, 0.0, 1.0)


def clip_mf(mf: np.ndarray, alpha: float) -> np.ndarray:
    return np.minimum(mf, alpha)


@dataclass(frozen=True)
class Rule:
    impacto: str
    urgencia: str
    tempo_aberto: str
    prioridade: str


class FuzzyTicketPriority:
    """Sistema fuzzy Mamdani para priorização de tickets de suporte."""

    def __init__(self) -> None:
        self.x_impacto = np.linspace(0, 10, 1001)
        self.x_urgencia = np.linspace(0, 10, 1001)
        self.x_tempo = np.linspace(0, 10, 1001)
        self.x_prioridade = np.linspace(0, 100, 2001)

        self.impacto = {
            "baixo": trapmf(self.x_impacto, 0, 0, 2.5, 4.0),
            "medio": trimf(self.x_impacto, 2.5, 5.0, 7.5),
            "alto": trapmf(self.x_impacto, 6.0, 8.0, 10.0, 10.0),
        }
        self.urgencia = {
            "baixa": trapmf(self.x_urgencia, 0, 0, 2.5, 4.0),
            "media": trimf(self.x_urgencia, 2.5, 5.0, 7.5),
            "alta": trapmf(self.x_urgencia, 6.0, 8.0, 10.0, 10.0),
        }
        self.tempo_aberto = {
            "curto": trapmf(self.x_tempo, 0, 0, 1.8, 3.5),
            "medio": trimf(self.x_tempo, 2.5, 5.0, 7.5),
            "longo": trapmf(self.x_tempo, 6.0, 8.0, 10.0, 10.0),
        }
        self.prioridade = {
            "baixa": trapmf(self.x_prioridade, 0, 0, 20, 40),
            "media": trimf(self.x_prioridade, 30, 50, 70),
            "alta": trimf(self.x_prioridade, 60, 75, 90),
            "critica": trapmf(self.x_prioridade, 80, 90, 100, 100),
        }

        self.rules = [
            Rule("alto", "alta", "longo", "critica"),
            Rule("alto", "alta", "curto", "alta"),
            Rule("alto", "media", "longo", "alta"),
            Rule("alto", "baixa", "longo", "alta"),
            Rule("medio", "alta", "longo", "alta"),
            Rule("medio", "alta", "medio", "alta"),
            Rule("medio", "media", "medio", "media"),
            Rule("medio", "baixa", "curto", "media"),
            Rule("baixo", "alta", "longo", "media"),
            Rule("baixo", "media", "medio", "media"),
            Rule("baixo", "baixa", "longo", "baixa"),
            Rule("baixo", "baixa", "curto", "baixa"),
        ]

    def _interp(self, universe: np.ndarray, mf: np.ndarray, value: float) -> float:
        return float(np.interp(value, universe, mf))

    def evaluate(self, impacto: float, urgencia: float, tempo_aberto: float) -> dict:
        mu_impacto = {name: self._interp(self.x_impacto, mf, impacto) for name, mf in self.impacto.items()}
        mu_urgencia = {name: self._interp(self.x_urgencia, mf, urgencia) for name, mf in self.urgencia.items()}
        mu_tempo = {name: self._interp(self.x_tempo, mf, tempo_aberto) for name, mf in self.tempo_aberto.items()}

        activated = []
        aggregated = np.zeros_like(self.x_prioridade)
        for rule in self.rules:
            alpha = min(mu_impacto[rule.impacto], mu_urgencia[rule.urgencia], mu_tempo[rule.tempo_aberto])
            consequent = clip_mf(self.prioridade[rule.prioridade], alpha)
            aggregated = np.maximum(aggregated, consequent)
            activated.append({"rule": rule, "alpha": alpha})

        area = np.trapz(aggregated, self.x_prioridade)
        if area == 0:
            crisp = 0.0
        else:
            crisp = float(np.trapz(self.x_prioridade * aggregated, self.x_prioridade) / area)

        return {
            "inputs": {"impacto": impacto, "urgencia": urgencia, "tempo_aberto": tempo_aberto},
            "membership": {"impacto": mu_impacto, "urgencia": mu_urgencia, "tempo_aberto": mu_tempo},
            "rules": activated,
            "aggregated": aggregated,
            "priority": crisp,
        }

    def label_priority(self, value: float) -> str:
        if value < 35:
            return "baixa"
        if value < 60:
            return "media"
        if value < 80:
            return "alta"
        return "critica"

    def test_scenarios(self) -> list[dict]:
        cases = [
            (1.0, 1.0, 1.0, "baixa"),
            (3.0, 4.0, 3.0, "media"),
            (5.0, 5.0, 5.0, "media"),
            (8.5, 8.5, 9.5, "critica"),
            (9.0, 2.0, 7.5, "alta"),
            (2.5, 9.0, 8.0, "media"),
        ]
        results = []
        for impacto, urgencia, tempo, expected in cases:
            outcome = self.evaluate(impacto, urgencia, tempo)
            value = outcome["priority"]
            results.append(
                {
                    "impacto": impacto,
                    "urgencia": urgencia,
                    "tempo_aberto": tempo,
                    "saida": round(value, 2),
                    "classe": self.label_priority(value),
                    "esperado": expected,
                }
            )
        return results

    def decision_surface(self, tempo_aberto: float = 5.0, resolution: int = 41) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        xs = np.linspace(0, 10, resolution)
        ys = np.linspace(0, 10, resolution)
        grid = np.zeros((resolution, resolution), dtype=float)
        for i, impacto in enumerate(xs):
            for j, urgencia in enumerate(ys):
                grid[j, i] = self.evaluate(float(impacto), float(urgencia), float(tempo_aberto))["priority"]
        return xs, ys, grid

    def plot_memberships(self, output: Path) -> None:
        fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5))
        self._plot_terms(axes[0, 0], self.x_impacto, self.impacto, "Impacto")
        self._plot_terms(axes[0, 1], self.x_urgencia, self.urgencia, "Urgência")
        self._plot_terms(axes[1, 0], self.x_tempo, self.tempo_aberto, "Tempo em aberto")
        self._plot_terms(axes[1, 1], self.x_prioridade, self.prioridade, "Prioridade")
        fig.suptitle("Funções de pertinência do sistema fuzzy", fontsize=16, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(output, dpi=180, bbox_inches="tight")
        plt.close(fig)

    def _plot_terms(self, ax: plt.Axes, x: np.ndarray, terms: dict[str, np.ndarray], title: str) -> None:
        for name, mf in terms.items():
            ax.plot(x, mf, label=name)
        ax.set_title(title)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    def plot_decision_surface(self, output: Path, tempo_aberto: float = 5.0) -> None:
        xs, ys, grid = self.decision_surface(tempo_aberto=tempo_aberto)
        fig, ax = plt.subplots(figsize=(8.5, 6.5))
        mesh = ax.contourf(xs, ys, grid, levels=18, cmap="viridis")
        cbar = fig.colorbar(mesh, ax=ax)
        cbar.set_label("Prioridade crisp")
        ax.set_xlabel("Impacto")
        ax.set_ylabel("Urgência")
        ax.set_title(f"Superfície de decisão com tempo em aberto fixo em {tempo_aberto:.1f}")
        fig.tight_layout()
        fig.savefig(output, dpi=180, bbox_inches="tight")
        plt.close(fig)


NAVY = "#152238"
SLATE = "#39465E"
GOLD = "#B78A2A"
PAPER = "#F8F7F2"
LINE = "#D8D3C4"


def _set_pdf_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
            "axes.edgecolor": SLATE,
            "axes.labelcolor": SLATE,
            "xtick.color": SLATE,
            "ytick.color": SLATE,
        }
    )


def _page(fig_size=(8.27, 11.69), facecolor="white") -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=fig_size, facecolor=facecolor)
    ax.set_facecolor(facecolor)
    ax.axis("off")
    return fig, ax


def _add_report_header(ax: plt.Axes, title: str, page: int) -> None:
    ax.plot([0.06, 0.94], [0.94, 0.94], transform=ax.transAxes, color=NAVY, lw=0.9)
    ax.text(0.06, 0.955, "Sistemas de Controle Fuzzy", transform=ax.transAxes, fontsize=8.5, color=SLATE, va="bottom")
    ax.text(0.94, 0.955, f"p. {page}", transform=ax.transAxes, fontsize=8.5, color=SLATE, ha="right", va="bottom")
    ax.text(0.06, 0.91, title, transform=ax.transAxes, fontsize=15, fontweight="bold", color=NAVY, va="top")
    ax.plot([0.06, 0.94], [0.07, 0.07], transform=ax.transAxes, color=LINE, lw=0.7)
    ax.text(0.5, 0.045, "Gabriel Albuquerque, Davi Maciel e Alberto Acosta", transform=ax.transAxes, fontsize=8, color=SLATE, ha="center")


def _paragraph(ax: plt.Axes, x: float, y: float, text: str, width: int = 92, size: int = 9.5, color: str = "#111111") -> float:
    wrapped = fill(text, width=width)
    ax.text(x, y, wrapped, transform=ax.transAxes, fontsize=size, color=color, va="top", linespacing=1.35)
    return y - 0.035 - 0.022 * (wrapped.count("\n") + 1)


def _section(ax: plt.Axes, x: float, y: float, title: str) -> float:
    ax.text(x, y, title, transform=ax.transAxes, fontsize=11.5, fontweight="bold", color=NAVY, va="top")
    ax.plot([x, 0.94], [y - 0.015, y - 0.015], transform=ax.transAxes, color=GOLD, lw=0.8)
    return y - 0.04


def _styled_table(ax: plt.Axes, headers: list[str], rows: list[list[str]], bbox: list[float], font_size: float = 8.2) -> None:
    table = ax.table(cellText=rows, colLabels=headers, cellLoc="left", colLoc="left", loc="center", bbox=bbox)
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor(LINE)
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor(NAVY)
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F3F1EA")
        else:
            cell.set_facecolor("white")


def _academic_text_page(title: str, sections: list[tuple[str, str]], page: int) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = _page()
    _add_report_header(ax, title, page)
    y = 0.84
    for section_title, body in sections:
        y = _section(ax, 0.06, y, section_title)
        for paragraph in body.split("\n\n"):
            y = _paragraph(ax, 0.06, y, paragraph)
        y -= 0.015
    return fig, ax


def build_report_pdf(system: FuzzyTicketPriority, output: Path) -> None:
    from matplotlib.backends.backend_pdf import PdfPages

    _set_pdf_style()
    with PdfPages(output) as pdf:
        fig, ax = _page()
        ax.plot([0.12, 0.88], [0.88, 0.88], transform=ax.transAxes, color=NAVY, lw=1.3)
        ax.text(0.5, 0.82, "Priorização de tickets de suporte técnico\ncom lógica fuzzy", transform=ax.transAxes, ha="center", fontsize=20, fontweight="bold", color=NAVY, linespacing=1.2)
        ax.text(0.5, 0.72, "Gabriel Albuquerque, Davi Maciel e Alberto Acosta", transform=ax.transAxes, ha="center", fontsize=11.5)
        ax.text(0.5, 0.675, "CESUPA - Inteligência Artificial e Computacional", transform=ax.transAxes, ha="center", fontsize=10.5, color=SLATE)
        ax.text(0.5, 0.63, "Relatório técnico em formato acadêmico inspirado no padrão SBC", transform=ax.transAxes, ha="center", fontsize=9.5, color=SLATE)
        ax.text(0.16, 0.545, "Resumo", transform=ax.transAxes, fontsize=11.5, fontweight="bold", color=NAVY)
        resumo = (
            "Este trabalho apresenta um protótipo mínimo e reprodutível de priorização de tickets. "
            "A solução usa um sistema fuzzy Mamdani com três entradas, uma saída, doze regras, "
            "visualização das funções de pertinência, cenários de teste e análise de sensibilidade."
        )
        ax.text(0.16, 0.505, fill(resumo, width=78), transform=ax.transAxes, fontsize=9.8, va="top", linespacing=1.35)
        ax.text(0.16, 0.335, "Palavras-chave", transform=ax.transAxes, fontsize=10.5, fontweight="bold", color=NAVY)
        ax.text(0.16, 0.305, "Lógica fuzzy; Mamdani; priorização; suporte técnico; protótipo acadêmico.", transform=ax.transAxes, fontsize=9.5)
        ax.text(0.16, 0.22, "GitHub", transform=ax.transAxes, fontsize=10.5, fontweight="bold", color=NAVY)
        ax.text(0.16, 0.19, "https://github.com/Gaalbu/Fuzzy-sistem", transform=ax.transAxes, fontsize=9.5, color=SLATE)
        ax.plot([0.12, 0.88], [0.12, 0.12], transform=ax.transAxes, color=NAVY, lw=1.3)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = _academic_text_page(
            "1. Introdução e escopo",
            [
                (
                    "Problema e justificativa",
                    "O protótipo define a prioridade de atendimento de tickets de suporte técnico acadêmico quando as informações disponíveis são parcialmente subjetivas. A lógica fuzzy é adequada porque transforma avaliações graduais, como impacto baixo, urgência média e tempo longo, em uma decisão numérica interpretável.",
                ),
                (
                    "Modalidade e requisitos atendidos",
                    "Modalidade escolhida: Opção B, aplicação/produto baseado em controle fuzzy. O modelo possui 3 entradas, 1 saída, 12 regras efetivas, 6 cenários de teste, funções de pertinência visualizadas, manual de execução, código-fonte e declaração de uso de IA.",
                ),
                (
                    "Extensão opcional",
                    "Foi adotado o caminho mais simples da pontuação extra: levantamento bibliográfico complementar. Bases/termos de busca: Google Scholar, IEEE Xplore e Scopus; fuzzy control, Mamdani, TSK e fuzzy rule base. Referências-base: Zadeh (1965), Mamdani & Assilian (1975), Ross (2010) e Klir & Yuan (1995).",
                ),
            ],
            page=2,
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig = plt.figure(figsize=(11.5, 8.5))
        fig.patch.set_facecolor("white")
        gs = fig.add_gridspec(2, 2)
        axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
        system._plot_terms(axes[0], system.x_impacto, system.impacto, "Impacto")
        system._plot_terms(axes[1], system.x_urgencia, system.urgencia, "Urgência")
        system._plot_terms(axes[2], system.x_tempo, system.tempo_aberto, "Tempo em aberto")
        system._plot_terms(axes[3], system.x_prioridade, system.prioridade, "Prioridade")
        fig.suptitle("2. Funções de pertinência", fontsize=18, fontweight="bold", color=NAVY)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        headers = ["#", "Impacto", "Urgência", "Tempo", "Saída"]
        rows = [[str(i + 1), r.impacto, r.urgencia, r.tempo_aberto, r.prioridade] for i, r in enumerate(system.rules)]
        fig, ax = _page()
        _add_report_header(ax, "3. Base de regras", 4)
        _styled_table(ax, headers, rows, [0.08, 0.20, 0.84, 0.62], font_size=8.4)
        ax.text(0.08, 0.145, "Inferência: operador AND = mínimo; agregação = máximo; defuzzificação = centróide.", transform=ax.transAxes, fontsize=9.2, color=SLATE)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = _academic_text_page(
            "4. Implementação e reprodutibilidade",
            [
                ("Arquitetura", "O sistema foi mantido em um módulo principal, com um script separado apenas para gerar os artefatos. Essa decisão reduz dependências, facilita a leitura e deixa a execução reproduzível."),
                ("Estrutura", "src/fuzzy_system.py: implementação do modelo. src/build_artifacts.py: geração dos PDFs e figuras. docs/manual_execucao.md: instruções de execução. docs/declaracao_ia.md: declaração de uso de IA. artifacts/: relatório, apresentação e imagens."),
                ("Dependências", "A solução usa apenas Python, numpy e matplotlib, evitando bibliotecas fuzzy prontas para preservar transparência sobre fuzzificação, regras, agregação e defuzzificação."),
            ],
            page=5,
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        tests = system.test_scenarios()
        rows = [[f"{t['impacto']:.1f}", f"{t['urgencia']:.1f}", f"{t['tempo_aberto']:.1f}", f"{t['saida']:.2f}", t['classe'], t['esperado']] for t in tests]
        fig, ax = _page()
        _add_report_header(ax, "5. Experimentos", 6)
        _styled_table(ax, ["Impacto", "Urgência", "Tempo", "Saída", "Classe", "Esperado"], rows, [0.06, 0.36, 0.88, 0.42], font_size=8.6)
        ax.text(0.06, 0.28, fill("Os cenários cobrem casos baixos, médios, altos, fronteiriços e conflitantes. Os resultados mantêm coerência qualitativa: baixa prioridade em casos simples, prioridade alta em conflito de alto impacto e criticidade quando impacto, urgência e tempo são elevados.", width=96), transform=ax.transAxes, fontsize=9.4, color=SLATE, va="top")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11.5, 4.8))
        xs, ys, grid = system.decision_surface(tempo_aberto=5.0)
        mesh = ax.contourf(xs, ys, grid, levels=18, cmap="viridis")
        cbar = fig.colorbar(mesh, ax=ax)
        cbar.set_label("Prioridade crisp")
        ax.set_xlabel("Impacto")
        ax.set_ylabel("Urgência")
        ax.set_title("6. Sensibilidade da saída com tempo em aberto fixo em 5,0", color=NAVY, fontweight="bold")
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = _academic_text_page(
            "7. Conclusão, IA e referências",
            [
                ("Conclusão", "O protótipo atende ao escopo acadêmico, gera saída contínua, é reprodutível e permite defesa técnica. Como limitações, a base de regras pode ser ampliada e a validação pode ser aprofundada com dados reais."),
                ("Declaração de uso de IA", "GPT-5.5 da OpenAI foi usado como apoio na escrita, organização e revisão do código. O material final foi conferido manualmente antes da exportação."),
                ("Referências", "Zadeh (1965); Mamdani & Assilian (1975); Ross (2010); Klir & Yuan (1995)."),
            ],
            page=8,
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def build_presentation_pdf(system: FuzzyTicketPriority, output: Path) -> None:
    from matplotlib.backends.backend_pdf import PdfPages

    _set_pdf_style()
    slides = []

    def slide(title: str, subtitle: str | None = None) -> tuple[plt.Figure, plt.Axes]:
        fig, ax = plt.subplots(figsize=(13.333, 7.5), facecolor=PAPER)
        ax.set_facecolor(PAPER)
        ax.axis("off")
        ax.add_patch(plt.Rectangle((0, 0), 1, 0.07, transform=ax.transAxes, color=NAVY))
        ax.add_patch(plt.Rectangle((0, 0.07), 1, 0.012, transform=ax.transAxes, color=GOLD))
        ax.text(0.06, 0.89, title, transform=ax.transAxes, fontsize=24, fontweight="bold", color=NAVY, va="top")
        if subtitle:
            ax.text(0.06, 0.79, subtitle, transform=ax.transAxes, fontsize=15.5, color=SLATE, va="top")
        ax.text(0.94, 0.028, "Sistema Fuzzy Mamdani", transform=ax.transAxes, fontsize=9, color="white", ha="right", va="center")
        return fig, ax

    def card(ax: plt.Axes, x: float, y: float, w: float, h: float, title: str, body: str) -> None:
        ax.add_patch(plt.Rectangle((x, y), w, h, transform=ax.transAxes, facecolor="white", edgecolor=LINE, lw=1))
        ax.text(x + 0.03, y + h - 0.055, title, transform=ax.transAxes, fontsize=13.5, fontweight="bold", color=NAVY, va="top")
        text_width = max(30, int(w * 82))
        ax.text(x + 0.03, y + h - 0.12, fill(body, width=text_width), transform=ax.transAxes, fontsize=10.8, color="#222222", va="top", linespacing=1.2)

    fig, ax = slide("Priorização de tickets de suporte técnico", "Protótipo acadêmico com lógica fuzzy")
    ax.text(0.06, 0.56, "Gabriel Albuquerque | Davi Maciel | Alberto Acosta", transform=ax.transAxes, fontsize=16, color=NAVY)
    ax.text(0.06, 0.48, "Inteligência Artificial e Computacional - CESUPA", transform=ax.transAxes, fontsize=13, color=SLATE)
    ax.text(0.06, 0.34, "Apresentação de 6 a 7 minutos", transform=ax.transAxes, fontsize=12.5, color=SLATE)
    slides.append(fig)

    fig, ax = slide("Problema", "Filas de suporte exigem priorização sob incerteza qualitativa")
    card(ax, 0.06, 0.42, 0.40, 0.24, "Contexto", "Tickets chegam com informações incompletas sobre impacto, urgência e tempo em aberto.")
    card(ax, 0.54, 0.42, 0.40, 0.24, "Decisão", "O sistema apoia a ordenação de atendimentos com saída numérica e interpretável.")
    card(ax, 0.06, 0.17, 0.88, 0.16, "Adequação fuzzy", "A decisão é gradual e linguística; portanto, não deve ser tratada apenas como alta ou baixa.")
    slides.append(fig)

    fig, ax = slide("Modelo fuzzy", "Entradas, saída e mecanismo de inferência")
    card(ax, 0.06, 0.42, 0.25, 0.25, "Impacto", "0 a 10; baixo, médio e alto.")
    card(ax, 0.37, 0.42, 0.25, 0.25, "Urgência", "0 a 10; baixa, média e alta.")
    card(ax, 0.68, 0.42, 0.25, 0.25, "Tempo", "0 a 10; curto, médio e longo.")
    card(ax, 0.22, 0.16, 0.56, 0.18, "Saída", "Prioridade de 0 a 100; baixa, média, alta e crítica. Inferência Mamdani com centróide.")
    slides.append(fig)

    fig, ax = slide("Funções de pertinência", "Transições suaves preservam a interpretação do modelo")
    inset = fig.add_axes([0.12, 0.22, 0.76, 0.40])
    system._plot_terms(inset, system.x_prioridade, system.prioridade, "Prioridade")
    slides.append(fig)

    fig, ax = slide("Implementação", "Código enxuto, reproduzível e compatível com o relatório")
    card(ax, 0.06, 0.40, 0.40, 0.26, "Tecnologias", "Python, numpy e matplotlib. Sem biblioteca fuzzy pronta.")
    card(ax, 0.54, 0.40, 0.40, 0.26, "Reprodução", "python src/build_artifacts.py gera relatório, slides, gráficos e tabela de testes.")
    card(ax, 0.06, 0.16, 0.88, 0.14, "Organização", "README, manual, declaração de IA, código-fonte e artefatos finais em PDF.")
    slides.append(fig)

    fig, ax = slide("Resultados", "Seis cenários cobrem casos baixos, médios, altos, críticos e conflitantes")
    tests = system.test_scenarios()
    rows = [[f"{t['impacto']:.1f}", f"{t['urgencia']:.1f}", f"{t['tempo_aberto']:.1f}", f"{t['saida']:.2f}", t["classe"]] for t in tests]
    _styled_table(ax, ["Impacto", "Urgência", "Tempo", "Saída", "Classe"], rows, [0.08, 0.20, 0.84, 0.48], font_size=10)
    slides.append(fig)

    fig, ax = slide("Fechamento", "Síntese para defesa técnica")
    card(ax, 0.06, 0.42, 0.40, 0.23, "Atendido", "Modelo Mamdani funcional, 12 regras, 6 testes, gráficos e GitHub.")
    card(ax, 0.54, 0.42, 0.40, 0.23, "Limitações", "Validação pode ser ampliada com dados reais e consulta a usuários.")
    card(ax, 0.06, 0.17, 0.88, 0.16, "Uso de IA", "Declarado no relatório e no repositório: GPT-5.5 da OpenAI.")
    slides.append(fig)

    with PdfPages(output) as pdf:
        for fig in slides:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def main() -> None:
    base = Path(__file__).resolve().parents[1]
    dist = base / "artifacts"
    dist.mkdir(exist_ok=True)

    system = FuzzyTicketPriority()
    system.plot_memberships(dist / "funcoes_pertinencia.png")
    system.plot_decision_surface(dist / "superficie_decisao.png", tempo_aberto=5.0)
    build_report_pdf(system, dist / "relatorio_fuzzy.pdf")
    build_presentation_pdf(system, dist / "apresentacao_fuzzy.pdf")

    print("Arquivos gerados em artifacts/")
    for file in sorted(dist.iterdir()):
        print(f"- {file.name}")
    print("\nCenários de teste:")
    for test in system.test_scenarios():
        print(
            f"- impacto={test['impacto']:.1f}, urgencia={test['urgencia']:.1f}, tempo={test['tempo_aberto']:.1f} -> "
            f"{test['saida']:.2f} ({test['classe']})"
        )


if __name__ == "__main__":
    main()
