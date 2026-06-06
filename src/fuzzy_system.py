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


def _draw_wrapped(ax: plt.Axes, x: float, y: float, text: str, width: int = 98, size: int = 10, bold: bool = False) -> float:
    lines = fill(text, width=width)
    ax.text(x, y, lines, transform=ax.transAxes, fontsize=size, va="top", ha="left", weight="bold" if bold else "normal")
    return 0.04 + 0.018 * lines.count("\n")


def _page(fig_size=(8.27, 11.69)) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=fig_size)
    ax.axis("off")
    return fig, ax


def _table_page(title: str, headers: list[str], rows: list[list[str]], note: str | None = None) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = _page()
    ax.text(0.05, 0.95, title, transform=ax.transAxes, fontsize=16, fontweight="bold", va="top")
    table = ax.table(cellText=rows, colLabels=headers, cellLoc="left", colLoc="left", loc="center", bbox=[0.05, 0.18, 0.9, 0.68])
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    if note:
        ax.text(0.05, 0.08, fill(note, width=100), transform=ax.transAxes, fontsize=9, va="bottom")
    return fig, ax


def build_report_pdf(system: FuzzyTicketPriority, output: Path) -> None:
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(output) as pdf:
        # Capa
        fig, ax = _page()
        ax.text(0.5, 0.92, "RELATÓRIO TÉCNICO", transform=ax.transAxes, ha="center", fontsize=18, fontweight="bold")
        ax.text(0.5, 0.88, "Priorização de tickets de suporte técnico com lógica fuzzy", transform=ax.transAxes, ha="center", fontsize=16)
        ax.text(0.5, 0.82, "Disciplina: Inteligência Artificial e Computacional", transform=ax.transAxes, ha="center", fontsize=12)
        ax.text(0.5, 0.79, "Tema: Sistemas de Controle Fuzzy", transform=ax.transAxes, ha="center", fontsize=12)
        ax.text(0.5, 0.73, "Integrantes", transform=ax.transAxes, ha="center", fontsize=13, fontweight="bold")
        ax.text(0.5, 0.69, "Gabriel Albuquerque\nDavi Maciel\nAlberto Acosta", transform=ax.transAxes, ha="center", fontsize=12, linespacing=1.6)
        ax.text(0.5, 0.56, "Resumo", transform=ax.transAxes, ha="center", fontsize=13, fontweight="bold")
        resumo = (
            "Este trabalho apresenta um protótipo mínimo e reprodutível de priorização de tickets. "
            "A solução usa um sistema fuzzy Mamdani com três entradas, uma saída, doze regras, "
            "visualização das funções de pertinência, cenários de teste e análise de sensibilidade."
        )
        ax.text(0.5, 0.48, fill(resumo, width=82), transform=ax.transAxes, ha="center", fontsize=11, va="top")
        ax.text(0.5, 0.18, "GitHub: https://github.com/Gaalbu/Fuzzy-sistem", transform=ax.transAxes, ha="center", fontsize=10)
        ax.text(0.5, 0.12, "Entrega acadêmica preparada para o repositório GitHub do projeto.", transform=ax.transAxes, ha="center", fontsize=10)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Problema e requisitos
        fig, ax = _page()
        ax.text(0.05, 0.95, "1. Problema e justificativa", transform=ax.transAxes, fontsize=16, fontweight="bold", va="top")
        texto = (
            "Problema: definir, de forma transparente e consistente, a prioridade de atendimento de tickets de suporte "
            "com informações parcialmente subjetivas. A lógica fuzzy é adequada porque o decisor trabalha com termos "
            "linguísticos como baixo, médio, alto e crítico.\n\n"
            "Modalidade escolhida: Opção B, produto/protótipo.\n\n"
            "Requisitos mínimos atendidos: 3 entradas, 1 saída, 12 regras efetivas, 6 cenários de teste, gráficos, "
            "manual de execução, código organizado e declaração de uso de IA."
        )
        ax.text(0.05, 0.88, fill(texto, width=95), transform=ax.transAxes, fontsize=10, va="top")
        ax.text(0.05, 0.53, "2. Extensão opcional escolhida", transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")
        extra = (
            "Foi adotado o caminho mais simples da pontuação extra: levantamento bibliográfico complementar com bases "
            "reconhecidas e autores clássicos da área fuzzy, sem aumentar a complexidade do código principal.\n\n"
            "Bases/termos de busca: Google Scholar, IEEE Xplore e Scopus; fuzzy control, Mamdani, TSK, fuzzy rule base.\n"
            "Referências-base para esse reforço: Zadeh (1965), Mamdani & Assilian (1975), Ross (2010) e Klir & Yuan (1995)."
        )
        ax.text(0.05, 0.47, fill(extra, width=95), transform=ax.transAxes, fontsize=10, va="top")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Funções de pertinência
        fig = plt.figure(figsize=(11.5, 8.5))
        gs = fig.add_gridspec(2, 2)
        axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
        system._plot_terms(axes[0], system.x_impacto, system.impacto, "Impacto")
        system._plot_terms(axes[1], system.x_urgencia, system.urgencia, "Urgência")
        system._plot_terms(axes[2], system.x_tempo, system.tempo_aberto, "Tempo em aberto")
        system._plot_terms(axes[3], system.x_prioridade, system.prioridade, "Prioridade")
        fig.suptitle("3. Funções de pertinência", fontsize=16, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Regras
        headers = ["#", "Impacto", "Urgência", "Tempo", "Saída"]
        rows = [[str(i + 1), r.impacto, r.urgencia, r.tempo_aberto, r.prioridade] for i, r in enumerate(system.rules)]
        fig, ax = _table_page("4. Base de regras", headers, rows, "Inferência: operador AND = mínimo; agregação = máximo; defuzzificação = centróide.")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Implementação
        fig, ax = _page()
        ax.text(0.05, 0.95, "5. Implementação e organização", transform=ax.transAxes, fontsize=16, fontweight="bold", va="top")
        body = (
            "Arquitetura: um único módulo de regra e um script de geração de artefatos. O objetivo foi manter o código "
            "curto, legível e fácil de reproduzir.\n\n"
            "Estrutura do repositório:\n"
            "- src/fuzzy_system.py\n"
            "- src/build_artifacts.py\n"
            "- docs/manual_execucao.md\n"
            "- docs/declaracao_ia.md\n"
            "- docs/roteiro_gamma.md\n"
            "- artifacts/relatorio_fuzzy.pdf\n"
            "- artifacts/apresentacao_fuzzy.pdf"
        )
        ax.text(0.05, 0.87, fill(body, width=92), transform=ax.transAxes, fontsize=10, va="top")
        ax.text(0.05, 0.44, "Decisão técnica", transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")
        ax.text(0.05, 0.39, fill("Foi evitada qualquer dependência pesada: apenas numpy e matplotlib.", width=92), transform=ax.transAxes, fontsize=10, va="top")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Experimentos
        tests = system.test_scenarios()
        rows = [[f"{t['impacto']:.1f}", f"{t['urgencia']:.1f}", f"{t['tempo_aberto']:.1f}", f"{t['saida']:.2f}", t['classe'], t['esperado']] for t in tests]
        fig, ax = _table_page("6. Cenários de teste", ["Impacto", "Urgência", "Tempo", "Saída", "Classe", "Esperado"], rows, "Os cenários cobrem casos baixos, médios, altos, fronteiriços e conflitantes.")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11.5, 4.8))
        xs, ys, grid = system.decision_surface(tempo_aberto=5.0)
        mesh = ax.contourf(xs, ys, grid, levels=18, cmap="viridis")
        cbar = fig.colorbar(mesh, ax=ax)
        cbar.set_label("Prioridade crisp")
        ax.set_xlabel("Impacto")
        ax.set_ylabel("Urgência")
        ax.set_title("Sensibilidade da saída com tempo em aberto fixo em 5,0")
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Conclusão e IA
        fig, ax = _page()
        ax.text(0.05, 0.95, "7. Conclusão, IA e referências", transform=ax.transAxes, fontsize=16, fontweight="bold", va="top")
        conclusion = (
            "Conclusão: o protótipo atende ao escopo acadêmico, gera saída contínua, é reprodutível e permite defesa técnica. "
            "Como limitações, a base de regras pode ser ampliada e a validação pode ser aprofundada com dados reais.\n\n"
            "Declaração de IA: GPT-5.5 Pro da OpenAI foi usado como apoio na escrita, organização e revisão do código. "
            "O material final foi conferido manualmente antes da exportação.\n\n"
            "Referências-base: Zadeh (1965), Mamdani & Assilian (1975), Ross (2010), Klir & Yuan (1995)."
        )
        ax.text(0.05, 0.87, fill(conclusion, width=95), transform=ax.transAxes, fontsize=10, va="top")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def build_presentation_pdf(system: FuzzyTicketPriority, output: Path) -> None:
    from matplotlib.backends.backend_pdf import PdfPages

    slides = []

    fig, ax = plt.subplots(figsize=(13.333, 7.5))
    ax.axis("off")
    ax.text(0.5, 0.78, "Priorização de tickets de suporte técnico", ha="center", fontsize=24, fontweight="bold")
    ax.text(0.5, 0.69, "Sistema fuzzy Mamdani", ha="center", fontsize=18)
    ax.text(0.5, 0.55, "Gabriel Albuquerque | Davi Maciel | Alberto Acosta", ha="center", fontsize=15)
    ax.text(0.5, 0.45, "Inteligência Artificial e Computacional", ha="center", fontsize=13)
    slides.append(fig)

    fig, ax = plt.subplots(figsize=(13.333, 7.5))
    ax.axis("off")
    ax.text(0.06, 0.88, "Problema", fontsize=20, fontweight="bold")
    ax.text(0.06, 0.74, fill("Tickets de suporte chegam com sinais imprecisos. A lógica fuzzy ajuda a converter esse contexto em prioridade de atendimento.", width=72), fontsize=16)
    ax.text(0.06, 0.47, "Por que fuzzy?", fontsize=18, fontweight="bold")
    ax.text(0.06, 0.36, fill("Porque a decisão é linguística, gradual e não binária.", width=72), fontsize=15)
    slides.append(fig)

    fig, ax = plt.subplots(figsize=(13.333, 7.5))
    ax.axis("off")
    ax.text(0.06, 0.9, "Modelo", fontsize=20, fontweight="bold")
    ax.text(0.06, 0.77, "Entradas: impacto, urgência, tempo em aberto", fontsize=16)
    ax.text(0.06, 0.67, "Saída: prioridade", fontsize=16)
    ax.text(0.06, 0.53, "12 regras efetivas e inferência Mamdani", fontsize=16)
    slides.append(fig)

    fig, ax = plt.subplots(figsize=(13.333, 7.5))
    ax.axis("off")
    ax.text(0.06, 0.88, "Funções de pertinência", fontsize=20, fontweight="bold")
    ax.text(0.06, 0.77, fill("As funções foram desenhadas para manter transições suaves e interpretação fácil.", width=72), fontsize=15)
    inset = fig.add_axes([0.08, 0.1, 0.84, 0.48])
    system._plot_terms(inset, system.x_prioridade, system.prioridade, "Prioridade")
    slides.append(fig)

    fig, ax = plt.subplots(figsize=(13.333, 7.5))
    ax.axis("off")
    ax.text(0.06, 0.88, "Implementação", fontsize=20, fontweight="bold")
    ax.text(0.06, 0.75, "Python + numpy + matplotlib", fontsize=17)
    ax.text(0.06, 0.63, "Código curto, sem dependência pesada", fontsize=17)
    ax.text(0.06, 0.51, "Geração automática do relatório e da apresentação", fontsize=17)
    slides.append(fig)

    fig, ax = plt.subplots(figsize=(13.333, 7.5))
    ax.axis("off")
    ax.text(0.06, 0.88, "Resultados", fontsize=20, fontweight="bold")
    tests = system.test_scenarios()
    preview = "\n".join([f"{i+1}. {t['impacto']:.1f} / {t['urgencia']:.1f} / {t['tempo_aberto']:.1f} -> {t['saida']:.2f} ({t['classe']})" for i, t in enumerate(tests)])
    ax.text(0.06, 0.76, preview, fontsize=13, va="top", linespacing=1.7)
    slides.append(fig)

    fig, ax = plt.subplots(figsize=(13.333, 7.5))
    ax.axis("off")
    ax.text(0.06, 0.86, "Fechamento", fontsize=20, fontweight="bold")
    ax.text(0.06, 0.73, fill("O sistema cumpre o escopo acadêmico e pode ser ampliado futuramente com mais regras, dados reais e comparação com outros modelos.", width=72), fontsize=16)
    ax.text(0.06, 0.47, "Uso de IA declarado no documento", fontsize=16)
    ax.text(0.06, 0.39, "GPT-5.4 Mini da OpenAI", fontsize=16, fontweight="bold")
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
