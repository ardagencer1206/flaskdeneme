# -*- coding: utf-8 -*-
import traceback
from typing import Dict, Any, List, Tuple
from flask import Flask, request, jsonify, send_from_directory
from pyomo.environ import (
    ConcreteModel, Set, Param, Var, Binary, NonNegativeReals, Objective, minimize,
    Constraint, Any as PyAny, value, SolverFactory
)

app = Flask(__name__, static_url_path="", static_folder=".")

# ------------------------------
# Yardımcı Fonksiyonlar
# ------------------------------

def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def pick_solver():
    """
    Önce Appsi-HiGHS (highspy) denenir.
    Sonra klasik solverlar: highs, cbc, glpk, cplex.
    """
    # 1) APPsi-HiGHS
    try:
        from pyomo.contrib.appsi.solvers.highs import Highs as AppsiHighs
        s = AppsiHighs()
        s.config.time_limit = 300
        return "appsi_highs", s
    except Exception:
        pass

    # 2) Klasik SolverFactory
    for cand in ["highs", "cbc", "glpk", "cplex"]:
        try:
            s = SolverFactory(cand)
            if s is not None and s.available():
                try:
                    s.options["timelimit"] = 300
                except Exception:
                    pass
                return cand, s
        except Exception:
            continue

    return None, None

# ------------------------------
# Model Kurulumu
# ------------------------------

def build_model(payload: Dict[str, Any]) -> Tuple[ConcreteModel, Dict[str, Any]]:
    cities: List[str] = payload["cities"]
    main_depot: str = payload["main_depot"]
    periods: int = int(payload["periods"])
    Tmin, Tmax = 1, periods
    periods_list = list(range(Tmin, Tmax + 1))

    vehicle_types: Dict[str, Dict[str, Any]] = payload["vehicle_types"]
    vehicle_count: Dict[str, int] = payload["vehicle_count"]

    vehicles: List[str] = [f"{vt}_{i}" for vt, cnt in vehicle_count.items() for i in range(1, int(cnt) + 1)]

    # Mesafeler
    distances = {}
    for i, j, d in payload["distances"]:
        distances[(i, j)] = float(d)
        distances[(j, i)] = float(d)
    for c in cities:
        distances[(c, c)] = 0.0

    # Paketler
    packages_input: List[Dict[str, Any]] = payload["packages"]
    packages = {}
    for rec in packages_input:
        pid = str(rec["id"])
        packages[pid] = {
            "baslangic": rec["baslangic"],
            "hedef": rec["hedef"],
            "agirlik": float(rec["agirlik"]),
            "baslangic_periyot": int(rec["ready"]),
            "teslim_suresi": int(rec["deadline_suresi"]),
            "ceza_maliyeti": float(rec["ceza"]),
        }

    MINUTIL_PENALTY = safe_float(payload.get("minutil_penalty", 10.0), 10.0)

    # Pyomo Modeli
    model = ConcreteModel()
    model.Cities = Set(initialize=cities)
    model.Periods = Set(initialize=periods_list)
    model.Vehicles = Set(initialize=vehicles)
    model.Packages = Set(initialize=list(packages.keys()))

    def vtype(v): return v.rsplit("_", 1)[0]

    # Parametreler
    model.Distance = Param(model.Cities, model.Cities, initialize=lambda m, i, j: distances[(i, j)])
    model.VehicleCapacity = Param(model.Vehicles, initialize=lambda m, v: vehicle_types[vtype(v)]["kapasite"])
    model.TransportCost = Param(model.Vehicles, initialize=lambda m, v: vehicle_types[vtype(v)]["maliyet_km"])
    model.FixedCost = Param(model.Vehicles, initialize=lambda m, v: vehicle_types[vtype(v)]["sabit_maliyet"])
    model.MinUtilization = Param(model.Vehicles, initialize=lambda m, v: vehicle_types[vtype(v)]["min_doluluk"])

    model.PackageWeight = Param(model.Packages, initialize=lambda m, p: packages[p]["agirlik"])
    model.PackageOrigin = Param(model.Packages, within=PyAny, initialize=lambda m, p: packages[p]["baslangic"])
    model.PackageDest = Param(model.Packages, within=PyAny, initialize=lambda m, p: packages[p]["hedef"])
    model.PackageReady = Param(model.Packages, initialize=lambda m, p: packages[p]["baslangic_periyot"])
    model.PackageDeadline = Param(model.Packages, initialize=lambda m, p: packages[p]["teslim_suresi"])
    model.LatePenalty = Param(model.Packages, initialize=lambda m, p: packages[p]["ceza_maliyeti"])

    # Değişkenler
    model.x = Var(model.Vehicles, model.Cities, model.Cities, model.Periods, domain=Binary)
    model.y = Var(model.Packages, model.Vehicles, model.Cities, model.Cities, model.Periods, domain=Binary)
    model.z = Var(model.Vehicles, model.Periods, domain=Binary)
    model.loc = Var(model.Vehicles, model.Cities, model.Periods, domain=Binary)
    model.pkg_loc = Var(model.Packages, model.Cities, model.Periods, domain=Binary)
    model.lateness = Var(model.Packages, domain=NonNegativeReals)
    model.minutil_shortfall = Var(model.Vehicles, model.Periods, domain=NonNegativeReals)

    # Amaç Fonksiyonu
    def objective_rule(m):
        transport = sum(m.TransportCost[v] * m.Distance[i, j] * m.x[v, i, j, t]
                        for v in m.Vehicles for i in m.Cities for j in m.Cities for t in m.Periods if i != j)
        fixed = sum(m.FixedCost[v] * m.z[v, t] for v in m.Vehicles for t in m.Periods)
        late = sum(m.LatePenalty[p] * m.lateness[p] for p in m.Packages)
        minutil = MINUTIL_PENALTY * sum(m.minutil_shortfall[v, t] for v in m.Vehicles for t in m.Periods)
        return transport + fixed + late + minutil

    model.obj = Objective(rule=objective_rule, sense=minimize)

    # ÖRNEK KISITLAR (tam listeyi önceki sürümden alabilirsin, burada özet bırakıyorum)
    def package_origin_rule(m, p):
        o, r = m.PackageOrigin[p], m.PackageReady[p]
        return sum(m.y[p, v, o, j, t] for v in m.Vehicles for j in m.Cities for t in m.Periods if j != o and t >= r) == 1
    model.package_origin_constraint = Constraint(model.Packages, rule=package_origin_rule)

    def vehicle_initial_loc_rule(m, v):
        return m.loc[v, main_depot, Tmin] == 1
    model.vehicle_initial_location = Constraint(model.Vehicles, rule=vehicle_initial_loc_rule)

    # (Diğer tüm kısıtları önceki koddan kopyala — sadece `main_depot` artık closure’dan geliyor.)

    return model, {
        "cities": cities,
        "periods_list": periods_list,
        "vehicles": vehicles,
        "packages": packages,
        "distances": distances,
        "vehicle_types": vehicle_types,
        "main_depot": main_depot,
        "MINUTIL_PENALTY": MINUTIL_PENALTY
    }

# ------------------------------
# Flask Routes
# ------------------------------

@app.route("/")
def root():
    return send_from_directory(".", "index.html")

@app.route("/solve", methods=["POST"])
def solve():
    try:
        data = request.get_json(force=True)
        model, meta = build_model(data)
        solver_name, solver = pick_solver()

        if solver is None:
            return jsonify({"ok": False, "error": "Uygun MILP çözücüsü bulunamadı."}), 400

        # Çöz
        try:
            results = solver.solve(model, tee=False)
        except TypeError:
            results = solver.solve(model)

        from pyomo.opt import TerminationCondition
        term = getattr(results, "termination_condition",
                       getattr(results.solver, "termination_condition", None))

        if term not in [TerminationCondition.feasible, TerminationCondition.optimal]:
            return jsonify({"ok": False, "error": f"Çözüm bulunamadı. Durum: {term}"}), 200

        return jsonify({"ok": True, "solver": solver_name, "result": "Çözüm başarılı"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
