# Arctic-XBeach
Coupled thermo-morphodynamic model for Arctic permafrost coastlines, combining XBeach hydrodynamics with an enthalpy-based thermal erosion module

## Overview
Arctic-XBeach couples the XBeach morphodynamic model with an enthalpy-based thermal erosion module to simulate coastal retreat processes in permafrost environments. The model uses an innovative event-driven approach that executes erosion calculations only during storms when thawed sediment is available, improving computational efficiency while maintaining physical realism.

## Key Features
- **Coupled approach**: Integrates XBeach hydrodynamics with permafrost thaw processes
- **Enthalpy-based thermal module**: Robust handling of phase change in frozen sediments
- **Event-driven methodology**: Storm-focused calculations optimize computational performance
- **Arctic-specific**: Designed for permafrost coastlines experiencing rapid climate change

## Status
🚧 **In Development** - This model is currently under active development. A manuscript describing the model formulation and validation is in preparation for submission to Geoscientific Model Development (GMD).

## Case Study
The model has been developed and tested for Barter Island, Alaska, leveraging well-documented permafrost characteristics and observational datasets for validation.

## Requirements
- Python 3.x
- XBeach
- Standard scientific Python stack (NumPy, SciPy, etc.)
*(Detailed requirements and installation instructions will be provided upon release)*

## Citation
If you use Arctic-XBeach in your research, please cite:
> Nederhoff., et al. (in preparation). Arctic-XBeach: A Python-Based Thermo-Morphodynamic Model for Arctic Permafrost Coastal Erosion. *Geoscientific Model Development*.

## License
Arctic-XBeach is licensed under the GNU General Public License v3.0 - see the license file for details.

## Contact
For questions or collaboration inquiries, please contact:
- Kees Nederhoff - Deltares USA
- Kevin de Bruijn - Delft University of Technology
- Carola Seyfert - Stichting Deltares Netherlands

---
*This is a Deltares Research project*
