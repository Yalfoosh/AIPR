SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")

CLASS_NAME_TO_NAME = {
    "EulerIntegrator": "Eulerov postupak",
    "InverseEulerIntegrator": "Obrnuti eulerov postupak",
    "TrapezoidalIntegrator": "Trapezni postupak",
    "RungeKutta4Integrator": "Postupak Runge-Kutta 4. reda",
    "Predictor EulerIntegrator, Corrector InverseEulerIntegrator": "PE(CE)2".translate(
        SUP
    ),
    "Predictor EulerIntegrator, Corrector TrapezoidalIntegrator": "PECE",
}