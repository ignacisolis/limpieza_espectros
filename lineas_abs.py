def gaussian(wavelength, flux, R=28000):
    """funcion para la convolucion del espectro"""
    from astropy.convolution import Gaussian1DKernel, convolve
    import numpy as np 
    
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
    import matplotlib.pyplot as plt
    import glob
    import os 
    from astropy.io import fits
    import numpy as np
    import re 
    from pathlib import Path
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

    match = re.search(r'star(\d+)', nombre)

    if match:
        numero = match.group(1)                     # "numero"
        star_completo = "star" + numero             # "star*"
        
        print(f"Archivo: {nombre_completo}")
        print(f"Star completo: {star_completo}")
        print(f"Número: {numero}")
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
        "Y": (9750, 11499),
        "J": (11500, 13200),
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
        plt.savefig(f"/home/nacho/molecfit_test/espectro_{star_completo}_banda_{nombre_banda}_{fe_1}.pdf", dpi=300, bbox_inches='tight')   
        plt.show()


def lineas_pdf(fits_file, fei_file):
    from scipy.signal import find_peaks
    import numpy as np
    import matplotlib.pyplot as plt
    from astropy.io import fits
    import os 
    import re
    from pathlib import Path
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

    data = np.loadtxt(fei_file)
    wavelength = data[:,0]
    flux = gaussian(wavelength, data[:,1])

    nombre_completo = Path(fits_file).name          # nombre con extensión
    nombre = Path(fits_file).stem                   # nombre sin extensión

    match = re.search(r'star(\d+)', nombre)

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
    wavelength = wavelength[mask]
    flux = flux[mask]   

    # buscar mínimos (absorciones)
    # Usar 1-flux para encontrar mínimos como picos
    inverted_flux = 1 - flux
    peaks, properties = find_peaks(inverted_flux, height=0.2, prominence=0.05)
    
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
               c='red', lw=1, label= f'Espectro observado {numero}')

        # FeI
        ax.plot(wavelength[mask_synth], flux[mask_synth], 
               lw=1, c='black', label='Espectro FeI')

        ax.set_xlim(xmin, xmax)
        

        ax.set_ylabel("Flujo")
        ax.set_title(f"Fe I λ = {lam:.2f} Å (profundidad: {heights[i]:.3f})")
        ax.legend(loc='upper right', fontsize=8)
        
        # Agregar línea vertical en la posición de la línea
        ax.axvline(x=lam, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)

    axes[-1].set_xlabel("λ (Angstrom)")
    
    plt.tight_layout()
    
    # Guardar figura
    output_file = f"/home/nacho/molecfit_test/lineas_de_hierro_{fe_1}_{star_completo}.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')

    plt.show()
    print(f"Gráfico guardado en {output_file}")


