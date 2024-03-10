"""
This module defines the class wrapping the base model and composing the mixture model
likelihood from the components and subgroups in the data.
"""
# pylint: disable=logging-fstring-interpolation

import logging
import warnings
from typing import Any, Iterable

import lymph
import numpy as np
import pandas as pd
from lymph import diagnose_times, modalities, types
from lymph.utils import flatten, popfirst, unflatten_and_split

from lymixture.utils import RESP_COLS, T_STAGE_COL, join_with_resps, normalize

pd.options.mode.copy_on_write = True
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
logger = logging.getLogger(__name__)



class LymphMixture(
    diagnose_times.Composite,   # NOTE: The order of inheritance must be the same as the
    modalities.Composite,       #       order in which the respective __init__ methods
    types.Model,                #       are called.
):
    """Class that handles the individual components of the mixture model."""
    def __init__(
        self,
        model_cls: type = lymph.models.Unilateral,
        model_kwargs: dict[str, Any] | None = None,
        num_components: int = 2,
    ):
        """Initialize the mixture model.

        The mixture will be based on the given ``model_cls`` (which is instantiated with
        the ``model_kwargs``), and will have ``num_components``.
        """
        if model_kwargs is None:
            model_kwargs = {"graph_dict": {
                ("tumor", "T"): ["II", "III"],
                ("lnl", "II"): ["III"],
                ("lnl", "III"): [],
            }}

        if not issubclass(model_cls, lymph.models.Unilateral):
            raise NotImplementedError(
                "Mixture model only implemented for `Unilateral` model."
            )

        self._model_cls = model_cls
        self._model_kwargs = model_kwargs
        self._mixture_coefs = None

        self.subgroups: dict[str, model_cls] = {}
        self.components: list[model_cls] = self._init_components(num_components)

        diagnose_times.Composite.__init__(
            self,
            distribution_children=dict(enumerate(self.components)),
            is_distribution_leaf=False,
        )
        logger.info(
            f"Created LymphMixtureModel based on {model_cls} model with "
            f"{num_components} components."
        )


    def _init_components(self, num_components: int) -> list[Any]:
        """Initialize the component parameters and assignments."""
        if num_components < 2:
            raise ValueError(f"A mixture of {num_components} does not make sense.")

        components = []
        for _ in range(num_components):
            components.append(self._model_cls(**self._model_kwargs))

        return components


    @property
    def is_trinary(self) -> bool:
        """Check if the model is trinary."""
        if not (
            all(sub.is_trinary for sub in self.subgroups.values())
            == all(comp.is_trinary for comp in self.components)
        ):
            raise ValueError("Subgroups & components not all trinary or not all binary.")

        return self.components[0].is_trinary


    def _init_mixture_coefs(self) -> pd.DataFrame:
        nan_array = np.empty((len(self.components), len(self.subgroups)))
        nan_array[:] = np.nan
        return pd.DataFrame(
            nan_array,
            index=range(len(self.components)),
            columns=self.subgroups.keys(),
        )


    def get_mixture_coefs(
        self,
        component: int | None = None,
        subgroup: str | None = None,
        norm: bool = True,
    ) -> float | pd.Series | pd.DataFrame:
        """Get mixture coefficients for the given ``subgroup`` and ``component``.

        The mixture coefficients are sliced by the given ``subgroup`` and ``component``
        which means that if no subgroupd and/or component is given, multiple mixture
        coefficients are returned.

        If ``normalize`` is set to ``True``, the mixture coefficients are normalized
        along the component axis before being returned.
        """
        if getattr(self, "_mixture_coefs", None) is None:
            self._mixture_coefs = self._init_mixture_coefs()

        if norm:
            self.normalize_mixture_coefs()

        component = slice(None) if component is None else component
        subgroup = slice(None) if subgroup is None else subgroup
        return self._mixture_coefs.loc[component, subgroup]


    def set_mixture_coefs(
        self,
        new_mixture_coefs: float | np.ndarray,
        component: int | None = None,
        subgroup: str | None = None,
    ) -> None:
        """Assign new mixture coefficients to the model.

        As in :py:meth:`~get_mixture_coefs`, ``subgroup`` and ``component`` can be used
        to slice the mixture coefficients and therefore assign entirely new coefs to
        the entire model, to one subgroup, to one component, or to one component of one
        subgroup.
        """
        if getattr(self, "_mixture_coefs", None) is None:
            self._mixture_coefs = self._init_mixture_coefs()

        component = slice(None) if component is None else component
        subgroup = slice(None) if subgroup is None else subgroup
        self._mixture_coefs.loc[component, subgroup] = new_mixture_coefs


    def normalize_mixture_coefs(self) -> None:
        """Normalize the mixture coefficients to sum to one."""
        if getattr(self, "_mixture_coefs", None) is not None:
            self._mixture_coefs = normalize(self._mixture_coefs, axis=0)


    def repeat_mixture_coefs(
        self,
        t_stage: str | None = None,
        log: bool = False,
        subgroup: str | None = None,
    ) -> np.ndarray:
        """Repeat mixture coefficients.

        The result will match the number of patients with tumors of ``t_stage`` that
        are in the specified ``subgroup`` (or all if it is set to ``None``). The
        mixture coefficients are returned in log-space if ``log`` is set to ``True``.
        """
        res = np.empty(shape=(0, len(self.components)))

        if subgroup is not None:
            subgroups = {subgroup: self.subgroups[subgroup]}
        else:
            subgroups = self.subgroups

        for label, subgroup in subgroups.items():
            has_t_stage = subgroup.patient_data[T_STAGE_COL] == t_stage
            num_patients = has_t_stage.sum() if t_stage is not None else len(has_t_stage)
            res = np.vstack([
                res, np.tile(self.get_mixture_coefs(subgroup=label), (num_patients, 1))
            ])

        return np.log(res) if log else res


    def get_params(
        self,
        as_dict: bool = True,
        as_flat: bool = True,
    ) -> Iterable[float] | dict[str, float]:
        """Get the parameters of the mixture model.

        This includes both the parameters of the individual components and the mixture
        coefficients. If a dictionary is returned (i.e. if ``as_dict`` is set to
        ``True``), the components' parameters are nested under keys that simply
        enumerate them. While the mixture coefficients are returned under keys of the
        form ``<subgroup>from<component>_coef``.

        The parameters are returned as a dictionary if ``as_dict`` is True, and as
        an iterable of floats otherwise. The argument ``as_flat`` determines whether
        the returned dict is flat or nested.

        >>> graph_dict = {
        ...     ("tumor", "T"): ["II", "III"],
        ...     ("lnl", "II"): ["III"],
        ...     ("lnl", "III"): [],
        ... }
        >>> mixture = LymphMixture(
        ...     model_kwargs={"graph_dict": graph_dict},
        ...     num_components=2,
        ... )
        >>> mixture.get_params(as_dict=True)     # doctest: +NORMALIZE_WHITESPACE
        {'0_TtoII_spread': 0.0,
         '0_TtoIII_spread': 0.0,
         '0_IItoIII_spread': 0.0,
         '1_TtoII_spread': 0.0,
         '1_TtoIII_spread': 0.0,
         '1_IItoIII_spread': 0.0}
        """
        params = {}
        for c, component in enumerate(self.components):
            params[str(c)] = component.get_params(as_flat=as_flat)

            for label in self.subgroups:
                params[str(c)].update({f"{label}_coef": self.get_mixture_coefs(c, label)})

        if as_flat or not as_dict:
            return flatten(params)

        return params if as_dict else params.values()


    def set_params(self, *args: float, **kwargs: float) -> tuple[float]:
        """Assign new params to the component models.

        This includes both the spread parameters for the component's models (if
        provided as positional arguments, they are used up first), as well as the
        mixture coefficients for the subgroups.

        Note:
            After setting all parameters, the mixture coefficients are normalized and
            may thus not be the same as the ones provided in the arguments.

        >>> graph_dict = {
        ...     ("tumor", "T"): ["II", "III"],
        ...     ("lnl", "II"): ["III"],
        ...     ("lnl", "III"): [],
        ... }
        >>> mixture = LymphMixture(
        ...     model_kwargs={"graph_dict": graph_dict},
        ...     num_components=2,
        ... )
        >>> mixture.set_params(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
        (0.7,)
        >>> mixture.get_params(as_dict=True)   # doctest: +NORMALIZE_WHITESPACE
        {'0_TtoII_spread': 0.1,
         '0_TtoIII_spread': 0.2,
         '0_IItoIII_spread': 0.3,
         '1_TtoII_spread': 0.4,
         '1_TtoIII_spread': 0.5,
         '1_IItoIII_spread': 0.6}
        """
        kwargs, global_kwargs = unflatten_and_split(
            kwargs, expected_keys=[str(c) for c, _ in enumerate(self.components)],
        )

        for c, component in enumerate(self.components):
            component_kwargs = global_kwargs.copy()
            component_kwargs.update(kwargs.get(str(c), {}))
            args = component.set_params(*args, **component_kwargs)

            for label in self.subgroups:
                first, args = popfirst(args)
                value = component_kwargs.get(f"{label}_coef", first)
                self.set_mixture_coefs(value, component=c, subgroup=label)

        self.normalize_mixture_coefs()
        return args


    def get_resps(
        self,
        patient: int | None = None,
        subgroup: str | None = None,
        component: int | None = None,
        t_stage: str | None = None,
        norm: bool = True,
    ) -> float | pd.Series | pd.DataFrame:
        """Get the repsonsibility of a ``patient`` for a ``component``.

        The ``patient`` index enumerates all patients in the mixture model unless
        ``subgroup`` is given, in which case the index runs over the patients in the
        given subgroup.

        Omitting ``component`` or ``patient`` (or both) will return corresponding slices
        of the responsibility table.

        The ``filter_by`` argument can be used to filter the responsibility table by
        any ``filter_value`` in the patient data. Most commonly, this is used to filter
        the responsibilities by T-stage.
        """
        if subgroup is not None:
            resp_table = self.subgroups[subgroup].patient_data[RESP_COLS]
        else:
            resp_table = self.patient_data[RESP_COLS]

        if norm:
            # double transpose, because pandas has weird broadcasting behavior
            resp_table = normalize(resp_table.T, axis=0).T

        if t_stage is not None:
            idx = resp_table["t_stage"] == t_stage
            if patient is not None and not idx[patient]:
                raise ValueError(f"Patient {patient} does not have T-stage {t_stage}.")
            if patient is not None:
                idx = patient
        else:
            idx = slice(None) if patient is None else patient

        component = slice(None) if component is None else component
        return resp_table.loc[idx,component]


    def set_resps(
        self,
        new_responsibilities: float | np.ndarray,
        patient: int | None = None,
        subgroup: str | None = None,
        component: int | None = None,
    ) -> None:
        """Assign ``new_responsibilities`` to the model.

        They should have the shape ``(num_patients, num_components)``, where
        ``num_patients`` is either the total number of patients in the model or only
        the number of patients in the ``subgroup`` (if that argument is not ``None``)
        and summing them along the last axis should yield a vector of ones.

        Note that these responsibilities essentially become the latent variables
        of the model or the expectation values of the latent variables (depending on
        whether or not they are "hardened", see :py:meth:`.harden_responsibilities`).
        """
        if isinstance(new_responsibilities, pd.DataFrame):
            new_responsibilities = new_responsibilities.to_numpy()

        pat_slice = slice(None) if patient is None else patient
        comp_slice = (*RESP_COLS, slice(None) if component is None else component)

        if subgroup is not None:
            sub_data = self.subgroups[subgroup].patient_data
            sub_data.loc[pat_slice,comp_slice] = new_responsibilities
            return

        patient_idx = 0
        for subgroup in self.subgroups.values():
            sub_data = subgroup.patient_data
            patient_idx += len(sub_data)

            if patient is not None:
                if patient_idx > patient:
                    sub_data.loc[pat_slice,comp_slice] = new_responsibilities
                    break
            else:
                sub_resp = new_responsibilities[:len(sub_data)]
                sub_data.loc[pat_slice,comp_slice] = sub_resp
                new_responsibilities = new_responsibilities[len(sub_data):]


    def load_patient_data(
        self,
        patient_data: pd.DataFrame,
        split_by: tuple[str, str, str],
        **kwargs,
    ):
        """Split the ``patient_data`` into subgroups and load it into the model.

        This amounts to computing the diagnose matrices for the individual subgroups.
        The ``split_by`` tuple should contain the three-level header of the LyProX-style
        data. Any additional keyword arguments are passed to the
        :py:meth:`~lymph.models.Unilateral.load_patient_data` method.
        """
        self._mixture_coefs = None
        grouped = patient_data.groupby(split_by)

        for label, data in grouped:
            if label not in self.subgroups:
                self.subgroups[label] = self._model_cls(**self._model_kwargs)
            data = join_with_resps(
                data, num_components=len(self.components)
            )
            self.subgroups[label].load_patient_data(data, **kwargs)

        modalities.Composite.__init__(
            self,
            modality_children=self.subgroups,
            is_modality_leaf=False,
        )


    @property
    def patient_data(self) -> pd.DataFrame:
        """Return all patients stored in the individual subgroups."""
        return pd.concat([
            subgroup.patient_data for subgroup in self.subgroups.values()
        ], ignore_index=True)


    def patient_component_likelihoods(
        self,
        t_stage: str | None = None,
        log: bool = True,
    ) -> np.ndarray:
        """Compute the (log-)likelihood of all patients, given the components.

        The returned array has shape ``(num_components, num_patients)`` and contains
        the likelihood of each patient with ``t_stage`` under each component. If ``log``
        is set to ``True``, the likelihoods are returned in log-space.
        """
        if t_stage is None:
            t_stages = self.t_stages
        else:
            t_stages = [t_stage]

        llhs = np.empty(shape=(0, len(self.components)))
        for subgroup in self.subgroups.values():
            sub_llhs = np.empty(shape=(len(subgroup.patient_data), len(self.components)))
            for t in t_stages:
                # use the index to align the likelihoods with the patients
                t_idx = subgroup.patient_data[T_STAGE_COL] == t
                sub_llhs[t_idx] = np.stack([
                    comp.state_dist(t) @ subgroup.diagnose_matrix(t).T
                    for comp in self.components
                ], axis=-1)
            llhs = np.vstack([llhs, sub_llhs])

        return np.log(llhs) if log else llhs


    def patient_mixture_likelihoods(
        self,
        t_stage: str | None = None,
        log: bool = True,
        marginalize: bool = False,
    ) -> np.ndarray:
        """Compute the (log-)likelihood of all patients under the mixture model.

        This is essentially the (log-)likelihood of all patients given the individual
        components as computed by :py:meth:`.patient_component_likelihoods` , but
        weighted by the mixture coefficients. This means that the returned array when
        ``marginalize`` is set to ``False`` represents the unnormalized expected
        responsibilities of the patients for the components.

        If ``marginalize`` is set to ``True``, the likelihoods are summed
        over the components, effectively marginalizing the components out of the
        likelihoods and yielding the incomplete data likelihood per patient.
        """
        component_patient_likelihood = self.patient_component_likelihoods(t_stage, log)
        full_mixture_coefs = self.repeat_mixture_coefs(t_stage, log)

        if log:
            llh = full_mixture_coefs + component_patient_likelihood
        else:
            llh = full_mixture_coefs * component_patient_likelihood

        if marginalize:
            return np.logaddexp.reduce(llh, axis=1) if log else np.sum(llh, axis=1)

        return llh


    def _incomplete_data_likelihood(
        self,
        t_stage: str | None = None,
        log: bool = True,
    ) -> float:
        """Compute the incomplete data likelihood of the model."""
        llhs = self.patient_mixture_likelihoods(t_stage, log, marginalize=True)
        return np.sum(llhs) if log else np.prod(llhs)


    def _complete_data_likelihood(
        self,
        t_stage: str | None = None,
        log: bool = True,
    ) -> float:
        """Compute the complete data likelihood of the model."""
        llhs = self.patient_mixture_likelihoods(t_stage, log)
        resps = self.get_resps(t_stage=t_stage).to_numpy()
        return np.sum(resps * llhs) if log else np.prod(llhs ** resps)


    def likelihood(
        self,
        given_params: Iterable[float] | dict[str, float] | None = None,
        given_resps: np.ndarray | None = None,
        log: bool = True,
    ) -> float:
        """Compute the (in-)complete data likelihood of the model.

        The likelihood is computed for the ``given_params``. If no parameters are given,
        the currently set parameters of the model are used.

        If responsibilities for each patient and component are given via ``given_resps``,
        they are used to compute the complete data likelihood. Otherwise, the incomplete
        data likelihood is computed, which marginalizes over the responsibilities.

        The likelihood is returned in log-space if ``log`` is set to ``True``.
        """
        try:
            # all functions and methods called here should raise a ValueError if the
            # given parameters are invalid...
            if given_params is None:
                pass
            elif isinstance(given_params, dict):
                self.set_params(**given_params)
            else:
                self.set_params(*given_params)
        except ValueError:
            return -np.inf if log else 0.

        if given_resps is not None:
            self.set_resps(given_resps)
            return self._complete_data_likelihood(log=log)

        return self._incomplete_data_likelihood(log=log)


    def risk(self):
        raise NotImplementedError
