"""
Estratègies d'staking (mida de l'aposta) per als backtests de ROI.

Kelly fraccionat: mida de l'aposta proporcional a l'edge (via la fórmula
de Kelly), moderada per una fracció perquè el model no està perfectament
calibrat -- Kelly complet és molt sensible a que prob_model sigui la
probabilitat real; si el model va una mica sobreconfiat, Kelly complet
sobredimensiona les apostes i dispara el risc de ruïna. Amb 1/4 o 1/2
Kelly es manté la idea de "apostar més quan hi ha més edge" sense
arriscar tant per errors de calibratge.
"""


def kelly_fraction(prob_model: float, quota: float) -> float:
    """
    Fracció òptima de Kelly (del bankroll) per apostar, donada la
    probabilitat que el model dona a l'opció i la quota decimal amb què
    es liquidaria l'aposta.

        f* = (b*p - q) / b
          b = quota - 1   (guany net per euro apostat si guanyes)
          p = prob_model
          q = 1 - p

    Si no hi ha edge real (f* <= 0), retorna 0.
    """
    b = quota - 1.0
    if b <= 0:
        return 0.0
    p = prob_model
    q = 1.0 - p
    f = (b * p - q) / b
    return max(f, 0.0)


def kelly_stake(prob_model: float, quota: float, bankroll: float, fraction: float = 0.25,
                 min_stake: float = 0.0, max_stake: float | None = None) -> float:
    """
    Stake en euros amb Kelly fraccionat: `fraction` de la Kelly completa
    (p.ex. 0.25 = quart de Kelly), aplicada sobre `bankroll`.

    min_stake/max_stake permeten evitar apostes irrisòries o
    desproporcionades; si el resultat és 0 (sense edge), sempre retorna 0
    encara que min_stake sigui > 0.
    """
    f = kelly_fraction(prob_model, quota)
    if f <= 0:
        return 0.0

    stake = f * fraction * bankroll
    if max_stake is not None:
        stake = min(stake, max_stake)
    return max(stake, min_stake)
