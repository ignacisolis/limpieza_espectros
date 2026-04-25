import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
from astropy.io import fits
import glob
import os
import re
import subprocess
from pathlib import Path
from astropy.convolution import Gaussian1DKernel, convolve
from macroturbulence import apply_vmac

def gaussian(wavelength, flux, R=28000):
    """funcion para la convolucion del espectro"""
    
    # considerar que la resolucion de winered en wide mode es de 28000

    # lambda central 
    lambda_central = np.median(wavelength)

    # calcular la FWHM en angstroms
    FWHM = lambda_central / R

    # calcular sigma en pixeles del espectro sintetico
    cdelt = 0.1
    sigma = (FWHM/cdelt)/2.355
    print(f"FWHM: {FWHM:.2f} Å, Sigma: {sigma:.2f} pixels")
    
    # crear el kernel gaussiano
    kernel = Gaussian1DKernel(sigma)

    # convolucion del espectro con el kernel gaussiano
    flux_conv = convolve(flux, kernel, boundary='extend')

    return flux_conv
def leer_lineas_hierro(path, tolerancia=0.5):
    """
    Lee el archivo de líneas de Fe I y retorna un diccionario
    con λ como clave y EP como valor
    """
    lineas = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            # Saltar headers y líneas vacías
            if not line or line.startswith("'"):
                continue
            try:
                cols = line.split()
                lam = float(cols[0])
                ep  = float(cols[1])
                loggf = float(cols[2])
                lineas.append({'lambda': lam, 'EP': ep, 'loggf': loggf})
            except (ValueError, IndexError):
                continue
    return lineas

def match_lineas(lineas_observadas, lineas_lista, tolerancia=0.5):
    """
    Para cada línea observada, busca la línea de Fe I más cercana
    en la lista dentro de la tolerancia dada (en Angstrom)
    
    Retorna lista con lambda, EP, loggf y distancia del match
    """
    resultados = []
    
    for lam_obs in lineas_observadas:
        mejor_match = None
        mejor_dist  = np.inf
        
        for linea in lineas_lista:
            dist = abs(lam_obs - linea['lambda'])
            if dist < mejor_dist and dist < tolerancia:
                mejor_dist  = dist
                mejor_match = linea
        
        if mejor_match is not None:
            resultados.append({
                'lambda_obs':   lam_obs,
                'lambda_lista': mejor_match['lambda'],
                'delta_lambda': mejor_match['lambda'] - lam_obs,
                'EP':           mejor_match['EP'],
                'loggf':        mejor_match['loggf'],
            })
            print(f"λ_obs={lam_obs:.2f} → λ_lista={mejor_match['lambda']:.3f} "
                  f"(Δ={mejor_match['lambda']-lam_obs:+.3f} Å) "
                  f"EP={mejor_match['EP']:.3f} eV  loggf={mejor_match['loggf']:.3f}")
        else:
            print(f"λ_obs={lam_obs:.2f} → sin match dentro de ±{tolerancia} Å")
    
    return resultados


def crear_lineas_pdf(ruta_observado, teff=4800,logg=2.7,Metalicidad = -1.3,V_macro = 3.75,rango_metalicidad=[-2.0, -1.0], microturb = 2.5 , espaciado = 0.1):
    """Función para crear los modelos sintéticos y comparar los peaks del espectro sintético
    con el observado, para luego plotear y guardar en un pdf
    
    Parámetros: 
    rango_metalicidad: lista con el rango de metalicidades a considerar, el intervalo será de 0.1 dex
    ruta_observado: ruta del espectro observado en formato fits
    patron_sintetico: patrón para encontrar los espectros sintéticos"""

    #%%%%%%%%%%%%%%%%%%%%%%% generar modelos atmosfericos %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #compilar el wrapper con el nombre del archivo output
    
    ruta_wrapper = "/home/nacho/molecfit_test/sintesis/marcs/pymarcs"
    directorio_trabajo = os.path.dirname(ruta_wrapper)
    
    ruta_modelos_turbospectrum = "/home/nacho/molecfit_test/Turbospectrum2019/COM-v19.1/models"
    # verificar si el modelo ya existe en Turbospectrum
    output_file= f'modelo_Gan1_{teff}_{logg}_{Metalicidad}.int'
    if os.path.exists(os.path.join(ruta_modelos_turbospectrum, output_file)):
        print(f"El modelo {output_file} ya existe en {ruta_modelos_turbospectrum}. No se generará nuevamente.")
    else:
        result = subprocess.run(
            [ruta_wrapper, str(teff), str(logg), str(Metalicidad), '--outroot', f'modelo_Gan1_{teff}_{logg}_{Metalicidad}.'],
            cwd=directorio_trabajo, # Asegúrate de que esta variable esté definida
            check=True,
            capture_output=True,
            text=True
        )
        print("Modelo generado con éxito")
    
    
    output_path = os.path.join(directorio_trabajo, output_file)
    destino_modelo = os.path.join(ruta_modelos_turbospectrum, output_file)
    if os.path.exists(destino_modelo):
        print(f"El modelo {output_file} ya existe en {ruta_modelos_turbospectrum}. No se moverá el archivo.")
    else:
        try:
            os.rename(output_path, destino_modelo)
            print(f"Archivo movido a {destino_modelo}")
        except Exception as e:
            print(f"Error al mover el archivo: {e}")

    #%%%%%%%%%%%%%%%%%%%%%%% generación de espectros sintéticos %%%%%%%%%%%%%%%%%%%%%%%
    ruta_archivos_sinteticos = "/home/nacho/molecfit_test/Turbospectrum2019/COM-v19.1/syntspec"
    patron_sintetico = f"modelo_Gan1_{teff}_{logg}_{Metalicidad}*.spec"
    archivos_existenes = glob.glob(os.path.join(ruta_archivos_sinteticos, patron_sintetico))
    # convertir a str la turbulencia
    microturb_str = f"{microturb:.1f}"
    # espaciado = float(input("Ingrese el espaciado entre metalicidades (ej: 0.1): "))
    # generar lista
    M_H_range = np.arange(rango_metalicidad[0], rango_metalicidad[1] + 0.1, espaciado)
    print(f"Rango de metalicidades a procesar: {M_H_range}")
    path_codigo = "/home/nacho/molecfit_test/Turbospectrum2019/COM-v19.1/scrip-star-gran1-flux.com"
    # nombre de referencia: modelo_Gan1_4800_2.5_-1.3.int_-1.30_9749.868-13189.917_xit3.0_todo.spec
    M_H_faltantes = []
    for M_H in M_H_range:
        M_H_str = f"{M_H:+.2f}"
        expected_files = glob.glob(os.path.join(ruta_archivos_sinteticos, f"modelo_Gan1_{teff}_{logg}_{Metalicidad}.int_{M_H_str}_*xit{microturb_str}*.spec"))
        if not expected_files:
            M_H_faltantes.append(M_H)
    
    if not M_H_faltantes:
        print("Todos los archivos sintéticos para el rango de metalicidades ya existen.")
    else:
        print(f"Faltan los siguientes archivos sintéticos para las metalicidades y microturbulencia: {M_H_faltantes}, {microturb_str}")
        print("Se procederá a generar los archivos faltantes.")
        
        with open(path_codigo, 'r') as file:
            lines = file.readlines()

        for M_H in M_H_faltantes:
            M_H_str = f"{M_H:+.2f}"
            
            print(f"Procesando metalicidad: {M_H_str} y microturbulencia: {microturb_str}")
            new_lines = []
            for line in lines:
                if "foreach MODEL"in line:
                    line = re.sub(r"(foreach MODEL\s*\()[^)]*(\))", 
                                rf"\g<1>{output_file}\2", line)
                elif re.match(r"^\s*set METALLIC\s*=", line):
                    line = re.sub(r"(set METALLIC\s*=\s*')[^']*(')", 
                                rf"\g<1>{M_H_str}\2", line)
                elif re.match(r"^\s*set TURBVEL\s*=", line):
                    line = re.sub(r"(set TURBVEL\s*=\s*')[^']*(')", 
                                rf"\g<1>{microturb_str}\2", line)
                new_lines.append(line)

            directorio = os.path.dirname(path_codigo)
            temp_path = os.path.join(directorio, f"temp_{M_H_str}.com")
            print(f"Usando archivo temporal: {temp_path}")

            with open(temp_path, 'w') as temp_file:
                temp_file.writelines(new_lines)

            os.chmod(temp_path, 0o755)

            try:
                result = subprocess.run(
                    ["csh", temp_path],
                    cwd=directorio,
                    check=True,
                    capture_output=True,
                    text=True
                )
                print(f"--- OUTPUT ({M_H_str}) ---")
                print(result.stdout)
                if result.stderr:
                    print(f"--- ERRORES ({M_H_str}) ---")
                    print(result.stderr)
            except subprocess.CalledProcessError as e:
                print(f"Error ejecutando {temp_path}: {e}")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    print(f"Archivo temporal eliminado: {temp_path}")
    
    #%%%%%%%%%%%%%% procesamiento de espectros %%%%%%%%%%%%%%%%%%%%%%%
    
    if not os.path.exists(ruta_archivos_sinteticos):
        raise FileNotFoundError(f"El directorio {ruta_archivos_sinteticos} no existe")
    
    archivos_sinteticos = glob.glob(os.path.join(ruta_archivos_sinteticos, patron_sintetico))
    
    if len(archivos_sinteticos) == 0:
        raise FileNotFoundError(f"No se encontraron archivos que coincidan con el patrón {patron_sintetico}")
    print(f"Archivos sintéticos encontrados: {len(archivos_sinteticos)}")
    for a in archivos_sinteticos:
        print(f" - {a}")
    
    # Leer espectro observado
    with fits.open(ruta_observado, mode='readonly', memmap=False) as hdul:
        if len(hdul) < 2:
            raise ValueError(f"El archivo FITS no tiene la extensión esperada (HDUs: {len(hdul)})")
        data_fits = hdul[1].data
        flujo_fits = data_fits['CFLUX']
        wavelength_fits = data_fits['LAMBDA'] * 1e4
    
    print(f"Espectro observado de {ruta_observado} leído correctamente")
    
    # Cargar espectros sintéticos
    espectros = []
    for file in archivos_sinteticos:
        data = np.loadtxt(file)
        wavelength = data[:,0]
        flux= data[:,1]
        flux = apply_vmac(wavelength, flux, V_macro, debug=False)  # Aplicar macroturbulencia
        flux = gaussian(wavelength, flux, R=28000)
        m = re.search(r'int_([+-]?\d+\.\d+)', file)
        if m:
            MH = float(m.group(1))
            espectros.append({'MH': MH, 'wavelength': wavelength, 'flux': flux})
    
    # Ordenar por metalicidad
    espectros = sorted(espectros, key=lambda x: x["MH"])
    
    # Detectar líneas en el espectro sintético de referencia (metalicidad media)
    idx_referencia = len(espectros) // 2
    flujo_sintetico = espectros[idx_referencia]['flux']
    wavelength_sintetico = espectros[idx_referencia]['wavelength']
    inverted_flux_synth = 1 - flujo_sintetico
    peaks_synth, properties_synth = find_peaks(inverted_flux_synth, height=0.3, prominence=0.05)
    lineas_obs = wavelength_sintetico[peaks_synth]
    
    window = 5  # angstroms alrededor de cada línea
    
    # Extraer nombre de la estrella
    nombre_estrella = Path(ruta_observado).stem
    match_star = re.search(r'_stars(\d+)', nombre_estrella)
    
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%% PDF comparativo %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    from matplotlib.backends.backend_pdf import PdfPages
    
    lineas_FE_path = "/home/nacho/molecfit_test/Turbospectrum2019/COM-v19.1/linelists/Fe_lines.list"
    lista_lineas =leer_lineas_hierro(lineas_FE_path)
    # hacer match entre lineas_obs y lista_lineas
    lineas_match = match_lineas(lineas_obs, lista_lineas, tolerancia=0.5)
    match_dict = {m['lambda_obs']: m for m in lineas_match}
    
    with PdfPages(f"{match_star.group(0)}_abs_segunMH.pdf") as pdf:
        for i, lam in enumerate(lineas_obs):
            fig = plt.figure(figsize=(10, 4))
            ax = fig.add_subplot(111)
            xmin, xmax = lam - window, lam + window
            
            mask_obs = (wavelength_fits >= xmin) & (wavelength_fits <= xmax)
            if np.any(mask_obs):

                prof_obs = 1 - np.min(flujo_fits[mask_obs])
                ax.plot(wavelength_fits[mask_obs], flujo_fits[mask_obs],
                       c='black', lw=2.5, alpha=1,zorder=5 ,label=f"Observado profundidad: {prof_obs:.3f}")
            
            # Plotear cada espectro sintético
            for esp in espectros:
                w = esp['wavelength']
                f = esp['flux']
                MH = esp['MH']
                mask_synth = (w >= xmin) & (w <= xmax)
                if np.any(mask_synth):
                    prof_syn = 1 - np.min(f[mask_synth])
                    ax.plot(w[mask_synth], f[mask_synth], lw=1, zorder=1,alpha=0.5,
                           label=f"[M/H]={MH:.2f} prof: {prof_syn:.3f}")
            
            ax.axvline(x=lam, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
            ax.set_xlim(xmin, xmax)
            ax.set_xlabel("λ (Angstrom)")
            ax.set_ylabel("Flujo")

            if lam in match_dict:
                info = match_dict[lam]
                titulo = (f"λ={lam:.2f} Å  "
                        f"EP={info['EP']:.3f} eV  Fe I "
                        f"(Δλ={info['delta_lambda']:+.3f} Å) "
                        f"V_macro={V_macro} km/s  "
                        f"Microturb={microturb} km/s "
                        f"Teff={teff} K  logg={logg}")
                ax.set_title(titulo, fontsize=11)
            else:
                ax.set_title(f"Línea observada en λ={lam:.2f} Å")
            ax.legend(fontsize=7, loc='lower left')
            pdf.savefig(fig)
            plt.close()
    
    print(f"PDF guardado como {match_star.group(0)}_abs_segunMH.pdf\n")
    
    #%%%%%%%%%%%%%%%%%%%%%% chi² %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    # Interpolar espectros sintéticos a la grilla observada
    print("Interpolando espectros sintéticos a la grilla observada...")
    espectros_interp = []
    for esp in espectros:
        interp_func = interp1d(esp['wavelength'], esp['flux'], 
                               bounds_error=False, fill_value=1.0)
        flux_interp = interp_func(wavelength_fits)
        espectros_interp.append({
            'MH': esp['MH'],
            'flux': flux_interp,
            'wavelength': wavelength_fits
        })
    
    # Analizar cada línea con chi^2 completo
    lineas_analizadas = []
    
    for idx, lam in enumerate(lineas_obs):
        mask = (wavelength_fits >= lam - window) & (wavelength_fits <= lam + window)
        
        if not np.any(mask):
            print(f"Advertencia: No hay datos para λ={lam:.2f}Å")
            continue
        
        # Datos observados en la región
        flujo_obs_region = flujo_fits[mask]
        wavelength_region = wavelength_fits[mask]
        
        # Calcular chi^2 para cada metalicidad
        chi2_values = []
        for esp in espectros_interp:
            flujo_synth_region = esp['flux'][mask]
            
            
            # Chi^2 
            residuales = (flujo_obs_region - flujo_synth_region) #/ sigma
            chi2 = np.sum(residuales**2 / flujo_obs_region)  
            # Profundidad sintética
            prof_synth = 1 - np.min(flujo_synth_region)
            
            chi2_values.append({
                'MH': esp['MH'],
                'chi2': chi2,
                'prof_synth': prof_synth,
                'flujo_synth': flujo_synth_region,
                'wavelength': wavelength_region
            })
        
        # Encontrar mejor ajuste por chi^2
        mejor_linea = min(chi2_values, key=lambda x: x['chi2'])
        
        prof_obs = 1 - np.min(flujo_obs_region)
        
        lineas_analizadas.append({
            'lambda': lam,
            'profundidad_obs': prof_obs,
            'mejor_MH': mejor_linea['MH'],
            'chi2_minimo': mejor_linea['chi2'],
            'prof_synth': mejor_linea['prof_synth'],
            'todas_las_MH': chi2_values,
            'flujo_obs_region': flujo_obs_region,
            'wavelength_region': wavelength_region
        })
        
        print(f"Línea {idx+1}: λ={lam:.2f}Å - Mejor [M/H]={mejor_linea['MH']:.3f} "
              f"(χ²={mejor_linea['chi2']:.4f}) - Prof_obs={prof_obs:.3f} - Prof_synth={mejor_linea['prof_synth']:.3f}")
    
    # Agregar análisis global combinando todas las líneas 

    if not lineas_analizadas:
        print("No se pudieron analizar líneas espectrales")
        return None
    
    # Método 1: Promedio simple
    MH_simple = np.mean([linea['mejor_MH'] for linea in lineas_analizadas])
    
    # Método 2: ponderación por varianza inversa
    pesos = [1.0 / (linea['chi2_minimo'] ) for linea in lineas_analizadas]
    MH_ponderado = np.average([linea['mejor_MH'] for linea in lineas_analizadas], weights=pesos)
    
    # Método 3: Chi^2 global sumando todas las líneas
    mh_unico = np.unique([mh for linea in lineas_analizadas 
                         for mh in [item['MH'] for item in linea['todas_las_MH']]])
    
    chi2_global = []
    for mh in mh_unico:
        chi2_total = 0
        g_libertad = 0
        for linea in lineas_analizadas:
            for item in linea['todas_las_MH']:
                if abs(item['MH'] - mh) < 0.05:
                    chi2_total += item['chi2']
                    g_libertad += 1
                    break
        if g_libertad > 0:
            chi2_global.append({'MH': mh, 'chi2_total': chi2_total/g_libertad}) # / g_libertad
    
    mejor_MH_global = min(chi2_global, key=lambda x: x['chi2_total'])['MH'] if chi2_global else MH_simple
    
    # Incertidumbres
    error_ponderado = np.sqrt(np.average([(linea['mejor_MH'] - MH_ponderado)**2 
                                         for linea in lineas_analizadas], weights=pesos) / len(lineas_analizadas))
    

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%% resultados %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    print("\n" + "="*60)
    print("RESULTADOS DE METALICIDAD")
    print("="*60)
    print(f"Metalicidad media (promedio simple):       {MH_simple:.3f} ± {np.std([l['mejor_MH'] for l in lineas_analizadas]):.3f} dex")
    print(f"Metalicidad media (ponderada por 1/χ²):    {MH_ponderado:.3f} ± {error_ponderado:.3f} dex")
    print(f"Metalicidad global (mínimo χ² total):      {mejor_MH_global:.3f} dex")
    print(f"Número de líneas analizadas:               {len(lineas_analizadas)}")
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%% diagnostico %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    
    # Gráfico 2: Metalicidad por línea con errores
    ax2 = axes[0, 0]
    # Extraer datos
    longitudes_onda = [linea['lambda'] for linea in lineas_analizadas]  # Longitudes de onda reales
    mejores_mh = [linea['mejor_MH'] for linea in lineas_analizadas]
    
    
    # calculo de error standar error of the mean
    std_mh = np.std(mejores_mh)
    sem = std_mh / np.sqrt(len(mejores_mh))

    
    # ax2.errorbar(ex=lineas_num, y=mejores_mh, yerr=errores_mh, fmt='o', color='blue', ecolor='lightblue', elinewidth=2, capsize=4, label='Mejor [M/H] por línea')
    ax2.scatter(longitudes_onda, mejores_mh, color='blue', label='Mejor [M/H] por línea')

    ax2.axhline(y=MH_ponderado, color='blue', linestyle='--', linewidth=2, label=f'Ponderado: {MH_ponderado:.3f} error: {sem:.3f} dex')
    # ax2.fill_between([0, len(lineas_analizadas)+1], 
    #                  MH_ponderado - error_ponderado, MH_ponderado + error_ponderado, 
    #                  color='blue', alpha=0.2)
    ax2.set_xlabel('Número de línea espectral')
    ax2.set_ylabel('[M/H] (dex)')
    ax2.set_title('Metalicidad por línea individual')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    
    # Gráfico 3: Chi^2 global
    ax3 = axes[0, 1]
    if chi2_global:
        mh_global = [c['MH'] for c in chi2_global]
        chi2_global_vals = [c['chi2_total'] for c in chi2_global]
        ax3.plot(mh_global, chi2_global_vals, 'b-', linewidth=2, marker='o', markersize=6)
        ax3.axvline(x=mejor_MH_global, color='red', linestyle='--', linewidth=2, label=f'Mejor: {mejor_MH_global:.3f}, χ²={min(chi2_global_vals):.4f}')
        ax3.set_xlabel('[M/H] (dex)')
        ax3.set_ylabel('χ² reducido')
        ax3.set_title(f'χ² reducido vs metalicidad grados de libertad = {g_libertad}')
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)
    
    # Gráfico 4: Comparación profundidades
    ax4 = axes[1, 0]
    for linea in lineas_analizadas:
        ax4.scatter(linea['profundidad_obs'], linea['prof_synth'], 
                   s=50, c='red', alpha=0.6, edgecolors='black', linewidth=1)
        ax4.annotate(f"{linea['lambda']:.1f}", 
                    (linea['profundidad_obs'], linea['prof_synth']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Línea 1:1
    min_prof = min(min([l['profundidad_obs'] for l in lineas_analizadas]), 
                   min([l['prof_synth'] for l in lineas_analizadas]))
    max_prof = max(max([l['profundidad_obs'] for l in lineas_analizadas]), 
                   max([l['prof_synth'] for l in lineas_analizadas]))
    ax4.plot([min_prof, max_prof], [min_prof, max_prof], 'k--', alpha=0.5, label='1:1')
    ax4.set_xlabel('Profundidad observada')
    ax4.set_ylabel('Profundidad sintética (mejor ajuste)')
    ax4.set_title('Comparación de profundidades')
    ax4.legend(loc='best')
    ax4.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.hist([linea['mejor_MH'] for linea in lineas_analizadas], bins=np.arange(MH_ponderado-0.5, MH_ponderado+0.5, 0.1), color='cyan', edgecolor='black')
    ax.axvline(x=MH_ponderado, color='blue', linestyle='--', linewidth=2, label=f'Ponderado: {MH_ponderado:.3f}')
    ax.set_xlabel('[M/H] (dex)')
    ax.set_ylabel('Número de líneas')
    ax.set_title('Distribución de mejores [M/H] por línea')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(f"{match_star.group(0)}_diagnostico_MH.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nFigura de diagnóstico guardada como {match_star.group(0)}_diagnostico_MH.png")
    
    # %%%%%%%%%%%%%%%%%%%%% resultados finales %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    resultados = {
        'MH_simple': MH_simple,
        'MH_ponderado': MH_ponderado,
        'MH_global': mejor_MH_global,
        'error_ponderado': error_ponderado,
        'std': np.std([linea['mejor_MH'] for linea in lineas_analizadas]),
        'n_lineas': len(lineas_analizadas),
        'lineas_analizadas': lineas_analizadas,
        'chi2_global': chi2_global
    }
    
    print("\n" + "="*60)
    print("RESUMEN FINAL")
    print("="*60)
    print(f"Mejor estimación de metalicidad: {MH_ponderado:.3f} ± {error_ponderado:.3f} dex")
    print(f"Basado en {len(lineas_analizadas)} líneas espectrales")
    
    return resultados