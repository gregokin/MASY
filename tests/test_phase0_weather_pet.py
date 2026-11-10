"""Phase 0: PET registry/delegate, shared weather schema bound to the owning SYRUP library, latitude binding.

Physical values (humidity derivation, Hargreaves PET) are produced by `maple_syrup.weather_pet`; these tests
check that MASY delegates to it, binds its results and refuses truthfully. They are skipped with an explicit
reason when the library is not importable (see `_phase0_support.LIBRARY_REPORT`).
"""

from __future__ import annotations

import datetime as dt
import sys
import unittest

import _phase0_support as ph
from _support import TempDirCase

from masy.pet import PET_METHODS, PETError, library_origin, pet_method, weather_pet_library
from masy.scenario_io import IOLimits
from masy.weather import LatitudeBinding, WeatherError, WeatherSeries, latitude_binding_from_case_yaml


class PETRegistryTests(unittest.TestCase):
    def test_registry_is_declarative_and_truthful(self) -> None:
        self.assertEqual(sorted(PET_METHODS), ["ecohyd_hargreaves", "explicit", "penman_monteith"])
        hargreaves = pet_method("ecohyd_hargreaves")
        self.assertTrue(hargreaves.needs_latitude)
        self.assertEqual(set(hargreaves.required_fields), {"tmean_c", "tmin_c", "tmax_c"})
        explicit = pet_method("explicit")
        self.assertFalse(explicit.needs_latitude)
        self.assertEqual(set(explicit.required_fields), {"tmean_c", "pet_mm_day"})
        self.assertEqual(hargreaves.owner, "maple_syrup.weather_pet")
        with self.assertRaises(PETError) as ctx:
            pet_method("penman_monteith")
        self.assertEqual(ctx.exception.code, "pet_method_unimplemented")
        for name in ("hargreaves_fao56", "priestley_taylor", "", None):
            with self.subTest(name=name):
                with self.assertRaises(PETError) as ctx:
                    pet_method(name)
                self.assertEqual(ctx.exception.code, "pet_method_unsupported")

    def test_masy_pet_module_holds_no_physics_and_no_import_time_library(self) -> None:
        import masy.pet as module
        source = open(module.__file__, encoding="utf-8").read()
        for forbidden in ("import math", "math.", "sqrt", "exp(", "cos(", "numpy", "0.0023 *", "* 0.0023"):
            self.assertNotIn(forbidden, source)  # constants may be NAMED in notes, never computed with
        self.assertNotIn("numpy", sys.modules.get("masy.pet").__dict__)

    @ph.requires_library
    def test_library_delegate_reports_actual_origin(self) -> None:
        module = weather_pet_library()
        origin = library_origin(module)
        self.assertEqual(origin["module"], "maple_syrup.weather_pet")
        self.assertEqual(origin["file"], module.__file__)
        self.assertEqual(len(origin["sha256"]), 64)
        self.assertIn("ecohyd_hargreaves", origin["declared_methods"])
        self.assertIn("not a source-locked qualified runtime", origin["qualification"])


@ph.requires_library
class WeatherTests(TempDirCase):
    def assertWeather(self, code: str, doc, **kwargs) -> None:
        with self.assertRaises(WeatherError) as ctx:
            WeatherSeries.from_input(doc, **kwargs)
        self.assertEqual(ctx.exception.code, code, ctx.exception.detail)

    def test_declarations_are_closed(self) -> None:
        series = WeatherSeries.from_input(ph.weather(ph.WEEK))
        self.assertEqual(series.pet_method_name, "ecohyd_hargreaves")
        self.assertEqual((series.first_date.isoformat(), series.last_date.isoformat()), (ph.WEEK[0], ph.WEEK[-1]))
        self.assertEqual(series.library["module"], "maple_syrup.weather_pet")
        bad_unit = ph.weather(ph.WEEK)
        bad_unit["units"]["relative_humidity"] = "percent"
        self.assertWeather("unsupported_unit", bad_unit)
        bad_sampling = ph.weather(ph.WEEK)
        bad_sampling["sampling"] = "hourly"
        self.assertWeather("unsupported_declaration", bad_sampling)
        latitude_field = ph.weather(ph.WEEK)
        latitude_field["latitude_deg"] = 32.0  # latitude is MAPLE's case.yaml field, not a weather field
        self.assertWeather("invalid_keys", latitude_field)
        bad_prov = ph.weather(ph.WEEK, provenance="simplified_synthetic")
        self.assertWeather("unsupported_declaration", bad_prov)
        self.assertWeather("pet_method_unimplemented", ph.weather(ph.WEEK, method="penman_monteith"))
        self.assertWeather("pet_method_unsupported", ph.weather(ph.WEEK, method="hargreaves_fao56"))
        foreign = ph.weather(ph.WEEK)
        foreign["humidity"]["derivation"] = "fao56_daily_mean_saturation"  # formulas are not re-declared in MASY
        self.assertWeather("unsupported_declaration", foreign)

    def test_fields_required_by_method(self) -> None:
        explicit = WeatherSeries.from_input(ph.weather(ph.WEEK, method="explicit", extrema=False, humidity="none"))
        self.assertEqual(set(explicit.rows[0]) - {"derived"}, {"date", "tmean_c", "pet_mm_day"})  # no unused extrema
        pet = explicit.daily_pet(explicit.rows[0], None)  # no latitude needed
        self.assertEqual((pet["method"], pet["pet_mm_day"], pet["clamped"], pet["latitude_source"]), ("explicit", 2.5, False, None))
        self.assertEqual(pet["provenance"]["latitude_deg"], None)
        no_temp = ph.weather(ph.WEEK, method="explicit", extrema=False, humidity="none")
        for row in no_temp["rows"]:
            del row["tmean_c"]
        self.assertWeather("invalid_keys", no_temp)  # explicit PET still requires temperature
        self.assertWeather("extrema_required", ph.weather(ph.WEEK, extrema=False, humidity="none"))  # Hargreaves needs extrema
        self.assertWeather("extrema_required", ph.weather(ph.WEEK, method="explicit", extrema=False, humidity="rh"))
        with_pet = ph.weather(ph.WEEK)
        with_pet["rows"][0]["pet_mm_day"] = 3.0  # undeclared field under hargreaves
        self.assertWeather("invalid_keys", with_pet)
        supplied = WeatherSeries.from_input(ph.weather(ph.WEEK, tmean="supplied"))
        self.assertEqual(supplied.rows[0]["tmean_c"], 8.0)
        self.assertNotIn("tmean_c", supplied.rows[0]["derived"])
        derived = WeatherSeries.from_input(ph.weather(ph.WEEK))
        self.assertEqual(derived.rows[0]["tmean_c"], 8.0)
        self.assertIn("declared MASY convention", derived.rows[0]["derived"]["tmean_c"])
        bad_order = ph.weather(ph.WEEK)
        bad_order["rows"][2]["tmax_c"] = -5.0
        self.assertWeather("weather_library_refused", bad_order)  # the owning library refuses tmin > tmax
        radiation = WeatherSeries.from_input(ph.weather(ph.WEEK, radiation=True))
        self.assertEqual(radiation.rows[0]["radiation_mj_m2_day"], 12.0)

    def test_humidity_is_derived_once_by_the_owning_library(self) -> None:
        library = weather_pet_library()
        rh = WeatherSeries.from_input(ph.weather(ph.WEEK, humidity="rh"))
        row = rh.rows[0]
        reference = library.DailyWeather(date=dt.date(2027, 12, 28), tmin_c=1.0, tmean_c=8.0, tmax_c=15.0,
                                         relative_humidity_fraction=0.4, provenance="prescribed")
        self.assertEqual(row["derived"]["vapor_pressure_deficit_kpa"], reference.vapor_pressure_deficit_kpa)
        self.assertEqual(row["derived"]["saturation_vapor_pressure_mean_kpa"], reference.saturation_vapor_pressure_mean_kpa)
        self.assertEqual(row["derived"]["humidity_sources"], dict(reference.humidity_sources))
        self.assertEqual(row["derived"]["owner"], "maple_syrup.weather_pet")
        vpd = WeatherSeries.from_input(ph.weather(ph.WEEK, humidity="vpd"))
        self.assertEqual(vpd.rows[0]["derived"]["humidity_sources"]["rh_source"], "derived")
        consistent = ph.weather(ph.WEEK, humidity="both")
        for r in consistent["rows"]:
            es = library.DailyWeather(date=dt.date.fromisoformat(r["date"]), tmin_c=r["tmin_c"], tmean_c=(r["tmin_c"] + r["tmax_c"]) / 2,
                                      tmax_c=r["tmax_c"], relative_humidity_fraction=0.4).saturation_vapor_pressure_mean_kpa
            r["vpd_kpa"] = es * 0.6
        both = WeatherSeries.from_input(consistent)
        self.assertEqual(both.rows[0]["derived"]["humidity_sources"], {"rh_source": "supplied", "vpd_source": "supplied"})
        self.assertWeather("weather_library_refused", ph.weather(ph.WEEK, humidity="both"))  # 0.6 kPa vs rh 0.4 disagree
        self.assertWeather("weather_library_refused", ph.weather(ph.WEEK, humidity="rh", rows_extra={"relative_humidity_fraction": 40.0}))
        none = WeatherSeries.from_input(ph.weather(ph.WEEK, humidity="none"))
        self.assertEqual(none.rows[0]["derived"], {"tmean_c": "arithmetic_mean_of_tmin_tmax (declared MASY convention)"})
        from masy.pet import daily_weather_parameters
        if any(n in daily_weather_parameters(library) for n in ("humidity_consistency_kpa", "humidity_tolerance_kpa", "consistency_tolerance_kpa")):
            # the library exposes a tolerance: the declared value is passed through, the library still decides
            loose = WeatherSeries.from_input(ph.weather(ph.WEEK, humidity="both", tolerance=1.0))
            self.assertEqual(loose.consistency_tolerance_kpa, 1.0)
            self.assertWeather("weather_library_refused", ph.weather(ph.WEEK, humidity="both", tolerance=0.05))
        else:
            self.assertWeather("humidity_tolerance_unsupported", ph.weather(ph.WEEK, humidity="both", tolerance=0.05))

    def test_rows_contiguous_coverage_and_pet(self) -> None:
        self.assertWeather("weather_rows_not_contiguous", ph.weather(ph.WEEK[:3] + ph.WEEK[4:]))
        self.assertWeather("empty_weather", ph.weather(()))
        series = WeatherSeries.from_input(ph.weather(ph.WEEK))
        rows = series.coverage((dt.date(2027, 12, 31), dt.date(2028, 1, 1)))
        self.assertEqual([r["date"] for r in rows], ["2027-12-31", "2028-01-01"])
        with self.assertRaises(WeatherError) as ctx:
            series.coverage((dt.date(2028, 1, 7),))
        self.assertEqual(ctx.exception.code, "weather_coverage_missing")
        with self.assertRaises(WeatherError) as ctx:
            series.daily_pet(rows[0], None)
        self.assertEqual(ctx.exception.code, "latitude_required")
        pet = series.daily_pet(rows[1], ph.lat())
        library = weather_pet_library()
        reference = library.potential_demand(
            library.DailyWeather(date=dt.date(2028, 1, 1), tmin_c=5.0, tmean_c=12.0, tmax_c=19.0, relative_humidity_fraction=0.4,
                                 provenance="prescribed"), 32.6, "ecohyd_hargreaves")
        self.assertEqual(pet["pet_mm_day"], reference.pet_mm_day)
        self.assertEqual(pet["ra_equiv_mm_day"], reference.ra_equiv_mm_day)
        self.assertEqual(pet["provenance"]["day_of_year"], 1)
        self.assertEqual(pet["provenance"]["source"], dict(library.SOURCE_IDENTITY))
        self.assertEqual(pet["latitude_source"]["field"], "location.latitude_deg")
        self.assertEqual(pet["library"]["module"], "maple_syrup.weather_pet")
        record = series.record()
        self.assertEqual(record["pet"]["needs_latitude"], True)
        self.assertEqual(record["humidity"]["derivation"], "maple_syrup.weather_pet")
        self.assertEqual(len(series.digest), 64)

    def test_latitude_binding_from_actual_case_yaml_semantics(self) -> None:
        limits = IOLimits()
        good = self.tmp / "case.yaml"
        good.write_text("name: demo\nlocation:\n  latitude_deg: 32.6\n")
        binding = latitude_binding_from_case_yaml(good, limits)
        self.assertEqual((binding.latitude_deg, binding.source["field"]), (32.6, "location.latitude_deg"))
        self.assertEqual(binding.source["sha256"], ph.sha(good.read_bytes()))
        self.assertEqual(LatitudeBinding.from_record(binding.record()), binding)
        cases = (("name: demo\n", "latitude_unspecified"), ("location:\n", "latitude_null"),
                 ("location:\n  latitude_deg: null\n", "latitude_null"), ("location:\n  lat: 3\n", "latitude_invalid"),
                 ("location:\n  latitude_deg: 91\n", "latitude_invalid"), ("location:\n  latitude_deg: true\n", "latitude_invalid"),
                 ("location:\n  latitude_deg: .inf\n", "case_yaml_unreadable"))
        for text, code in cases:
            with self.subTest(code=code):
                path = self.tmp / f"{code}.yaml"
                path.write_text(text)
                with self.assertRaises(WeatherError) as ctx:
                    latitude_binding_from_case_yaml(path, limits)
                self.assertEqual(ctx.exception.code, code, ctx.exception.detail)
        with self.assertRaises(WeatherError):
            LatitudeBinding(0.0, {"kind": "scenario_default", "path": "x", "sha256": "0" * 64, "field": "location.latitude_deg"})


if __name__ == "__main__":
    unittest.main()
