from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import os 
import re
from pathlib import Path
from astropy.convolution import Gaussian1DKernel, convolve
import glob
import sys
import subprocess
from scipy.optimize import curve_fit


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


def plot_spectrum(archivo, paths):
    
    """funcion para plotear los espectros sinteticos y el espectro observado
    paths: lista de los espectros sinteticos
    archivo: archivo fits del espectro observado"""

    # Verificar que el archivo FITS existe antes de intentar abrirlo
    if not os.path.exists(archivo):
        raise FileNotFoundError(f"El archivo FITS {archivo} no existe")
    
    print(f"leyendo el espectro observado de {archivo}...")
    
    try:
        with fits.open(archivo, mode='readonly', memmap=False) as hdul:
            # Verificar que el HDU existe
            if len(hdul) < 2:
                raise ValueError(f"El archivo FITS no tiene la extensión esperada (HDUs: {len(hdul)})")
            
            data_fits = hdul[1].data
            flujo_fits = data_fits['CFLUX']
            wavelength_fits = data_fits['LAMBDA'] * 1e4
            
    except Exception as e:
        print(f"Error al leer el archivo FITS: {e}")
        raise
    
    # verificar que la ruta de los espectros sinteticos exista
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"El archivo {p} no existe")
    
    nombre_completo = Path(archivo).name          # nombre con extensión
    nombre = Path(archivo).stem                   # nombre sin extensión

    match = re.search(r'_stars(\d+)', nombre)

    if match:
        numero = match.group(1)                     # "numero"
        star_completo = "star" + numero             # "star*"
        
        print(f"Archivo: {nombre_completo}")
        print(f"Star completo: {star_completo}")
        print(f"Número: {numero}")
        print("Sí funcionó :v")
    else:
        print("No se encontró el patrón star + número")

    print(f"Espectro observado de la estrella nº{numero} leído correctamente")

    flujo_fits = np.clip(flujo_fits, None, 1.05)

    #extraer la metalicidad del nombre del archivo .spec
    m = re.search(r'int_([+-]?\d+\.\d+)', paths[0])
    if m:
        fe_1 = m.group(1)
    else:
        raise ValueError("Metallicidad no encontrada")
    # preparar espectros sintéticos
    spectra = []
    for p in paths:
        data = np.loadtxt(p)
        wavelength = data[:,0]
        flujo_sin_procesar = data[:,1]
        flux = gaussian(wavelength, flujo_sin_procesar, R=28000)  # cambiar la resolucion
        nombre = os.path.basename(p).split("_")[-1].replace(".spec","")
        spectra.append((wavelength, flux, nombre))

    # definicion de bandas Y y J en Angstroms
    bandas = {
        "Y": (np.min(wavelength_fits), 11499),
        "J": (11500, np.max(wavelength_fits)),
    }
    
    for nombre_banda, (banda_min, banda_max) in bandas.items():
        n_segments = max(1, int((banda_max - banda_min) // 300))
        edges = np.linspace(banda_min, banda_max, n_segments + 1)

        fig, axes = plt.subplots(n_segments, 1, figsize=(12, 4 * n_segments), sharey=True)
        if n_segments == 1:
            axes = [axes]

        for i in range(n_segments):
            ax = axes[i]
            xmin = edges[i]
            xmax = edges[i+1]

            # espectro observado
            # Filtrar datos observados en el rango para mejor rendimiento
            mask_fits = (wavelength_fits >= xmin) & (wavelength_fits <= xmax)
            ax.plot(wavelength_fits[mask_fits], flujo_fits[mask_fits], 
                   c='red', lw=1, label=f'estrella nº{numero}')

            # espectros sinteticos
            for wavelength, flux, nombre in spectra:
                mask_synth = (wavelength >= xmin) & (wavelength <= xmax)
                ax.plot(wavelength[mask_synth], flux[mask_synth], 
                       lw=1, label=nombre)

            ax.set_xlim(xmin, xmax)
            #ax.set_ylim(0.90, 1.01)
            ax.set_ylabel("Flujo")
            ax.legend(loc='upper right', fontsize=8)
            
            if i == 0:
                ax.set_title(f"Espectro {star_completo} — Banda {nombre_banda}")
            if i == n_segments - 1:
                ax.set_xlabel("λ (Angstrom)")

        plt.tight_layout()
        plt.style.use('dark_background')
        plt.savefig(f"/home/nacho/molecfit_test/espectro_{star_completo}_banda_{nombre_banda}_{fe_1}.pdf", dpi=300, bbox_inches='tight')   
        plt.show()


def lineas_pdf(fits_file, fei_file):
    def gausss(x,a,mu,sigma):
        return a * np.exp(-(x-mu)**2/(2*sigma**2))
    # Verificar que el archivo existe
    if not os.path.exists(fits_file):
        raise FileNotFoundError(f"El archivo FITS {fits_file} no existe")
    
    # Verificar que el archivo de FeI existe
    
    if not os.path.exists(fei_file):
        raise FileNotFoundError(f"El archivo de FeI {fei_file} no existe")
    # extraer la metalicidad del archivo .spec
    
    try:
        with fits.open(fits_file, mode='readonly', memmap=False) as hdul:
            # Verificar que el HDU existe
            if len(hdul) < 2:
                raise ValueError(f"El archivo FITS no tiene la extensión esperada (HDUs: {len(hdul)})")
            
            data_fits = hdul[1].data
            flujo_fits = data_fits['CFLUX']
            wavelength_fits = data_fits['LAMBDA'] * 1e4
            
    except Exception as e:
        print(f"Error al leer el archivo FITS: {e}")
        raise
    
    lam_0 = 10811.12 # linea de feI conocida
    window_fwhm = 3
    mask = (wavelength_fits >= lam_0 - window_fwhm) & (wavelength_fits <= lam_0 + window_fwhm)
    x = wavelength_fits[mask]
    y = 1 - flujo_fits[mask]

    p0=[y.max(), lam_0, 0.5]
    popt, _ = curve_fit(gausss, x, y, p0=p0)
    sigma = abs(popt[2])
    FWHM = 2.355 * sigma
    #R = lam_0 / FWHM
    R = 22000
    print(f"FWHM: {FWHM:.2f} Å, Resolución: {R:.0f}")

    data = np.loadtxt(fei_file)
    wavelength = data[:,0]
    flux = gaussian(wavelength, data[:,1],R=R) # cambiar resolucion

    nombre_completo = Path(fits_file).name          # nombre con extensión
    nombre = Path(fits_file).stem                   # nombre sin extensión

    match = re.search(r'_stars(\d+)', nombre)

    if match:
        numero = match.group(1)                     # "numero"
        star_completo = "star" + numero             # "star*"
        
        print(f"Archivo: {nombre_completo}")
        print(f"Star completo: {star_completo}")
        print(f"Número: {numero}")
    else:
        print("No se encontró el patrón star + número")

    minimo_lamb = wavelength_fits.min()
    maximo_lamb = wavelength_fits.max()
    
    # filtrar rango
    mask = (wavelength >= minimo_lamb) & (wavelength <= maximo_lamb)
    flux = flux[mask]
    wavelength = wavelength[mask]
    
    # buscar mínimos (absorciones)
    # Usar 1-flux para encontrar mínimos como picos
    inverted_flux = 1 - flux
    peaks, properties = find_peaks(inverted_flux, height=0.3, prominence=0.05)
    
    lineas_FeI = wavelength[peaks]
    heights = properties['peak_heights']

    print(f"Lineas de FeI encontradas: {len(lineas_FeI)}")
    m = re.search(r'int_([+-]?\d+\.\d+)', fei_file)
    if m:
        fe_1 = m.group(1)
    else:
        raise ValueError("Metallicidad no encontrada")
    if len(lineas_FeI) == 0:
        print("No se encontraron líneas de FeI con la altura especificada")
        return
    
    window = 5  # Å alrededor de cada línea

    # Crear figura con subplots
    fig, axes = plt.subplots(len(lineas_FeI), 1, figsize=(10, 2*len(lineas_FeI)), sharey=True)
    
    # Asegurar que axes es iterable incluso con una sola línea
    if len(lineas_FeI) == 1:
        axes = [axes]

    for i, lam in enumerate(lineas_FeI):
        ax = axes[i]

        xmin = lam - window
        xmax = lam + window

        # Filtrar datos observados en el rango
        mask_obs = (wavelength_fits >= xmin) & (wavelength_fits <= xmax)
        mask_synth = (wavelength >= xmin) & (wavelength <= xmax)
        
        # observado
        ax.plot(wavelength_fits[mask_obs], flujo_fits[mask_obs], 
               c='red', lw=1, label= f'Espectro observado {numero} R={R:.0f}')

        # FeI
        ax.plot(wavelength[mask_synth], flux[mask_synth], 
               lw=1, label=f'Espectro sintético (M/H={fe_1})')

        ax.set_xlim(xmin, xmax)
        

        ax.set_ylabel("Flujo")
        ax.set_title(f" λ = {lam:.2f} Å (profundidad: {heights[i]:.3f})")
        ax.legend(loc='upper right', fontsize=8)
        
        # Agregar línea vertical en la posición de la línea
        ax.axvline(x=lam, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)

    axes[-1].set_xlabel("λ (Angstrom)")
    
    plt.tight_layout()
    plt.style.use('dark_background')
    # Guardar figura
    output_file = f"/home/nacho/molecfit_test/lineas_de_hierro_{fe_1}_{star_completo}.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')

    plt.show()
    print(f"Gráfico guardado en {output_file}")



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


def crear_lineas_pdf(ruta_observado, patron_sintetico, rango_metalicidad=[-2.0, -1.0], microturb = 2.5 , espaciado = 0.1):
    """Función para crear los modelos sintéticos y comparar los peaks del espectro sintético
    con el observado, para luego plotear y guardar en un pdf
    
    Parámetros: 
    rango_metalicidad: lista con el rango de metalicidades a considerar, el intervalo será de 0.1 dex
    ruta_observado: ruta del espectro observado en formato fits
    patron_sintetico: patrón para encontrar los espectros sintéticos"""
    
    ruta_archivos_sinteticos = "/home/nacho/molecfit_test/Turbospectrum2019/COM-v19.1/syntspec"
    archivos_existenes = glob.glob(os.path.join(ruta_archivos_sinteticos, patron_sintetico))
    # convertir a str la turbulencia
    microturb_str = f"{microturb:.1f}"
    # espaciado = float(input("Ingrese el espaciado entre metalicidades (ej: 0.1): "))
    # generar lista
    M_H_range = np.arange(rango_metalicidad[0], rango_metalicidad[1] + 0.1, espaciado)
    print(f"Rango de metalicidades a procesar: {M_H_range}")
    path_codigo = "/home/nacho/molecfit_test/Turbospectrum2019/COM-v19.1/scrip-star-gran1-flux.com"
    # nombre de referencia: gran1_syn_modelo2.int_-0.79_9749.868-13189.917_xit6.0_todo.spec
    M_H_faltantes = []
    for M_H in M_H_range:
        M_H_str = f"{M_H:+.2f}"
        expected_files = glob.glob(os.path.join(ruta_archivos_sinteticos, f"gran1_syn_modelo2.int_{M_H_str}_*xit{microturb_str}*.spec"))
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
                if re.match(r"^\s*set METALLIC\s*=", line):
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
        flux = gaussian(wavelength, data[:,1], R=28000)
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
            ax.set_title(f"Línea observada en λ={lam:.2f} Å")
            ax.legend(fontsize=7, loc='best')
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
            
            # Estimación del error (1% del flujo + ruido de fondo)
            #sigma = 0.01 * flujo_obs_region + 0.001
            
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
        
        # Interpolación parabólica para refinar la metalicidad
        # if len(chi2_values) >= 3:
        #     mh_vals = np.array([x['MH'] for x in chi2_values])
        #     chi2_vals = np.array([x['chi2'] for x in chi2_values])
        #     idx_min = np.argmin(chi2_vals)
            
        #     if 0 < idx_min < len(mh_vals) - 1:
        #         # Ajuste cuadrático
        #         x = mh_vals[idx_min-1:idx_min+2]
        #         y = chi2_vals[idx_min-1:idx_min+2]
        #         coeffs = np.polyfit(x, y, 2)
        #         if coeffs[0] > 0:  # Parábola con mínimo
        #             mh_minimo = -coeffs[1] / (2 * coeffs[0])
        #             chi2_minimo = np.polyval(coeffs, mh_minimo)
        #             mejor_linea = {
        #                 'MH': mh_minimo,
        #                 'chi2': chi2_minimo,
        #                 'prof_synth': mejor_linea['prof_synth'],
        #                 'flujo_synth': mejor_linea['flujo_synth'],
        #                 'wavelength': mejor_linea['wavelength']
        #             }
        
        # Profundidad observada
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
    
    # Método 2: Promedio ponderado por inverso de chi^2
    pesos = [1.0 / (linea['chi2_minimo'] + 1e-6) for linea in lineas_analizadas]
    MH_ponderado = np.average([linea['mejor_MH'] for linea in lineas_analizadas], weights=pesos)
    
    # Método 3: Chi^2 global sumando todas las líneas
    mh_unico = np.unique([mh for linea in lineas_analizadas 
                         for mh in [item['MH'] for item in linea['todas_las_MH']]])
    
    chi2_global = []
    for mh in mh_unico:
        chi2_total = 0
        n_lineas = 0
        for linea in lineas_analizadas:
            for item in linea['todas_las_MH']:
                if abs(item['MH'] - mh) < 0.05:
                    chi2_total += item['chi2']
                    n_lineas += 1
                    break
        if n_lineas > 0:
            chi2_global.append({'MH': mh, 'chi2_total': chi2_total/ n_lineas }) # / n_lineas
    
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
    
    # Gráfico 1: χ² vs Metalicidad para cada línea
    #ax1 = axes[0, 0]
    # for linea in lineas_analizadas:
    #     mh_vals = [item['MH'] for item in linea['todas_las_MH']]
    #     chi2_vals = [item['chi2'] for item in linea['todas_las_MH']]
    #     ax1.plot(mh_vals, chi2_vals, 'o-', label=f"{linea['lambda']:.1f}Å", alpha=0.7, markersize=4)
    # ax1.set_xlabel('[M/H] (dex)')
    # ax1.set_ylabel('χ² reducido')
    # ax1.set_title('χ² por línea espectral')
    # ax1.axvline(x=mejor_MH_global, color='red', linestyle='--', linewidth=2, label=f'Global: {mejor_MH_global:.3f}')
    # ax1.legend(loc='best', fontsize=8)
    # ax1.grid(True, alpha=0.3)
    
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
        ax3.axvline(x=mejor_MH_global, color='red', linestyle='--', linewidth=2, label=f'Mejor: {mejor_MH_global:.3f}')
        ax3.set_xlabel('[M/H] (dex)')
        ax3.set_ylabel('χ² total promedio')
        ax3.set_title('Ajuste global - χ² combinado')
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
        
        
    

