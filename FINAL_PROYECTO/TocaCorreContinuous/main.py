import numpy as np
from train import entrenar, guardar_estadisticas, cargar_estadisticas
from visualize import visualizar
from collections import defaultdict
import os

# ==========================================
#   GUARDAR/CARGAR MODELO
# ==========================================

def guardar_modelo(Q_per, Q_eva, stats=None):
    """Guarda modelo y estadísticas"""
    try:
        np.save("Q_per.npy", dict(Q_per))
        np.save("Q_eva.npy", dict(Q_eva))
        print("\n✅ Modelo guardado exitosamente")
        print(f"   - Q_per.npy ({len(Q_per)} estados)")
        print(f"   - Q_eva.npy ({len(Q_eva)} estados)")

        if stats is not None:
            guardar_estadisticas(stats, "training_stats.npy")

        return True
    except Exception as e:
        print(f"\n❌ Error al guardar: {e}")
        return False


def cargar_modelo():
    """Carga modelo entrenado"""
    try:
        Q_per_dict = np.load("Q_per.npy", allow_pickle=True).item()
        Q_eva_dict = np.load("Q_eva.npy", allow_pickle=True).item()

        Q_per = defaultdict(lambda: np.zeros(9))
        Q_eva = defaultdict(lambda: np.zeros(9))

        Q_per.update(Q_per_dict)
        Q_eva.update(Q_eva_dict)

        print("\n✅ Modelo cargado correctamente")
        print(f"   Estados Perseguidor: {len(Q_per)}")
        print(f"   Estados Evasor: {len(Q_eva)}")
        
        return Q_per, Q_eva
    except FileNotFoundError:
        print("\n❌ No se encontró modelo entrenado")
        print("   Por favor entrena primero (opción 1)")
        return None, None
    except Exception as e:
        print(f"\n❌ Error al cargar: {e}")
        return None, None


def verificar_modelo_existe():
    """Verifica si existe un modelo guardado"""
    return os.path.exists("Q_per.npy") and os.path.exists("Q_eva.npy")


# ==========================================
#   CONFIGURACIÓN
# ==========================================

def configurar_entrenamiento():
    """Configura parámetros de entrenamiento"""
    print("\n" + "="*60)
    print("  ⚙️  CONFIGURACIÓN DE ENTRENAMIENTO")
    print("="*60)

    try:
        episodios = int(input("Número de episodios [5000]: ") or "5000")
        gamma = float(input("Gamma (descuento) [0.95]: ") or "0.95")
        alpha = float(input("Alpha (learning rate) [0.1]: ") or "0.1")

        print(f"\n✅ Configuración:")
        print(f"   Episodios: {episodios}")
        print(f"   Gamma: {gamma}")
        print(f"   Alpha: {alpha}\n")
        
        return episodios, gamma, alpha
    except:
        print("\n⚠️  Entrada inválida, usando valores por defecto\n")
        return 5000, 0.95, 0.1


def configurar_visualizacion():
    """Configura parámetros de visualización"""
    print("\n" + "="*60)
    print("  🎨 CONFIGURACIÓN DE VISUALIZACIÓN")
    print("="*60)

    try:
        episodios = int(input("Episodios a visualizar [∞ continuo]: ") or "9999")
        pasos = int(input("Pasos máximos por episodio [300]: ") or "300")
        velocidad = int(input("Velocidad (ms entre frames) [50]: ") or "50")
        tray = input("Mostrar trayectorias? (s/n) [s]: ").lower() or "s"

        print(f"\n✅ Configuración:")
        print(f"   Episodios: {'∞ continuo' if episodios > 1000 else episodios}")
        print(f"   Pasos máximos: {pasos}")
        print(f"   Velocidad: {velocidad}ms")
        print(f"   Trayectorias: {'Sí' if tray == 's' else 'No'}\n")

        return episodios, pasos, velocidad, (tray == "s")
    except:
        print("\n⚠️  Entrada inválida, usando valores por defecto\n")
        return 9999, 300, 50, True


def analizar_estadisticas():
    """Analiza y muestra estadísticas del entrenamiento"""
    stats = cargar_estadisticas()
    
    if stats is None:
        print("❌ No hay estadísticas disponibles\n")
        return
    
    print("\n" + "="*70)
    print("  📊 ANÁLISIS DE ESTADÍSTICAS DE ENTRENAMIENTO")
    print("="*70)
    
    if len(stats['recompensas_per']) > 0:
        print(f"\n📈 Recompensas:")
        print(f"   Perseguidor:")
        print(f"     - Promedio: {np.mean(stats['recompensas_per']):.1f}")
        print(f"     - Máximo: {np.max(stats['recompensas_per']):.1f}")
        print(f"     - Mínimo: {np.min(stats['recompensas_per']):.1f}")
        print(f"\n   Evasor:")
        print(f"     - Promedio: {np.mean(stats['recompensas_eva']):.1f}")
        print(f"     - Máximo: {np.max(stats['recompensas_eva']):.1f}")
        print(f"     - Mínimo: {np.min(stats['recompensas_eva']):.1f}")
    
    if len(stats['pasos_por_episodio']) > 0:
        print(f"\n🚶 Pasos por episodio:")
        print(f"   - Promedio: {np.mean(stats['pasos_por_episodio']):.1f}")
        print(f"   - Máximo: {np.max(stats['pasos_por_episodio'])}")
        print(f"   - Mínimo: {np.min(stats['pasos_por_episodio'])}")
    
    if len(stats['intercambios_por_episodio']) > 0:
        print(f"\n🔄 Intercambios de roles:")
        print(f"   - Promedio por episodio: {np.mean(stats['intercambios_por_episodio']):.2f}")
        print(f"   - Total acumulado: {np.sum(stats['intercambios_por_episodio'])}")
    
    if 'evaluaciones' in stats and len(stats['evaluaciones']) > 0:
        print(f"\n🎯 Evaluaciones (sin exploración):")
        for eval_data in stats['evaluaciones'][-5:]:
            print(f"   Ep {eval_data['episodio']}: "
                  f"{eval_data['capturas']} capturas, "
                  f"{eval_data['pasos']:.1f} pasos, "
                  f"{eval_data['distancia']:.2f} dist")
    
    print("\n" + "="*70 + "\n")


# ==========================================
#   MENÚ PRINCIPAL
# ==========================================

def main():
    """Menú principal del juego"""
    print("\n" + "="*70)
    print("      🎮 TOCA Y CORRE - APRENDIZAJE POR REFUERZO")
    print("        Estudiante: Ariel Huarachi Clemente")
    print("="*70)

    while True:
        # Verificar si existe modelo
        modelo_existe = verificar_modelo_existe()
        
        print("\n📋 MENÚ PRINCIPAL")
        print("-" * 50)
        print("  1️⃣  Entrenar agentes desde cero")
        print("  2️⃣  Visualizar agentes entrenados" + 
              (" ✅" if modelo_existe else " ⚠️  (requiere entrenar)"))
        print("  3️⃣  Entrenar + Visualizar (rápido)")
        print("  4️⃣  Analizar estadísticas" + 
              (" ✅" if os.path.exists("training_stats.npy") else " ⚠️"))
        print("  5️⃣  Continuar entrenamiento existente")
        print("  6️⃣  Salir")
        print("-" * 50)

        opcion = input("\n👉 Seleccione una opción: ").strip()

        # ============ OPCIÓN 1: ENTRENAR ============
        if opcion == "1":
            print("\n" + "="*70)
            print("  🏋️  MODO ENTRENAMIENTO DESDE CERO")
            print("="*70)

            if modelo_existe:
                confirmar = input("\n⚠️  Ya existe un modelo. ¿Sobrescribir? (s/n): ").lower()
                if confirmar != 's':
                    print("Operación cancelada\n")
                    continue

            usar = input("\n¿Configurar parámetros personalizados? (s/n) [n]: ").lower()
            if usar == "s":
                ep, gamma, alpha = configurar_entrenamiento()
            else:
                ep, gamma, alpha = 5000, 0.95, 0.1
                print(f"\nUsando configuración por defecto:")
                print(f"  {ep} episodios, γ={gamma}, α={alpha}\n")

            input("Presiona ENTER para iniciar entrenamiento...")
            
            print("\n🚀 Iniciando entrenamiento...\n")
            Q_per, Q_eva, stats = entrenar(
                q_episodes=ep, gamma=gamma, alpha=alpha, verbose=True
            )

            guardar_modelo(Q_per, Q_eva, stats)
            print("\n✅ ¡Entrenamiento completado exitosamente!\n")
            
            input("Presiona ENTER para continuar...")

        # ============ OPCIÓN 2: VISUALIZAR ============
        elif opcion == "2":
            print("\n" + "="*70)
            print("  🎬 MODO VISUALIZACIÓN")
            print("="*70)

            Q_per, Q_eva = cargar_modelo()
            if Q_per is None:
                input("\nPresiona ENTER para volver al menú...")
                continue

            usar = input("\n¿Configurar visualización? (s/n) [n]: ").lower()
            if usar == "s":
                ep, pasos, vel, tray = configurar_visualizacion()
            else:
                ep, pasos, vel, tray = 9999, 300, 50, True
                print("\nUsando configuración por defecto (modo continuo)\n")

            print("🎮 Iniciando visualización...")
            print("   Controles: ESPACIO=Pausar | R=Reiniciar | ESC=Salir\n")
            
            visualizar(Q_per, Q_eva, episodios=ep, pasos=pasos,
                      mostrar_trayectorias=tray, velocidad=vel)
            
            print("\n✅ Visualización finalizada\n")

        # ============ OPCIÓN 3: ENTRENAR + VISUALIZAR ============
        elif opcion == "3":
            print("\n" + "="*70)
            print("  🎯 MODO RÁPIDO: ENTRENAR + VISUALIZAR")
            print("="*70)

            ep = input("\nEpisodios de entrenamiento [3000]: ") or "3000"
            ep = int(ep)
            
            print(f"\n🚀 Fase 1: Entrenando {ep} episodios...\n")
            Q_per, Q_eva, stats = entrenar(q_episodes=ep, gamma=0.95, 
                                          alpha=0.1, verbose=True)
            guardar_modelo(Q_per, Q_eva, stats)

            print("\n✅ Entrenamiento completado")
            print("🎬 Fase 2: Iniciando visualización...\n")
            
            visualizar(Q_per, Q_eva, episodios=3, pasos=300,
                      mostrar_trayectorias=True, velocidad=60)

        # ============ OPCIÓN 4: ANALIZAR ============
        elif opcion == "4":
            analizar_estadisticas()
            input("Presiona ENTER para continuar...")

        # ============ OPCIÓN 5: CONTINUAR ENTRENAMIENTO ============
        elif opcion == "5":
            print("\n" + "="*70)
            print("  ♻️  CONTINUAR ENTRENAMIENTO EXISTENTE")
            print("="*70)

            Q_per, Q_eva = cargar_modelo()
            if Q_per is None:
                print("\n⚠️  No hay modelo previo. Usa opción 1 para entrenar desde cero\n")
                input("Presiona ENTER para continuar...")
                continue

            ep = input("\nEpisodios adicionales [2000]: ") or "2000"
            ep = int(ep)
            
            print(f"\n🚀 Continuando entrenamiento por {ep} episodios...\n")
            
            # Nota: Al continuar, usar alpha más bajo
            Q_per, Q_eva, stats = entrenar(
                q_episodes=ep, gamma=0.95, alpha=0.05, verbose=True
            )
            
            guardar_modelo(Q_per, Q_eva, stats)
            print("\n✅ Entrenamiento adicional completado\n")
            
            input("Presiona ENTER para continuar...")

        # ============ OPCIÓN 6: SALIR ============
        elif opcion == "6":
            print("\n" + "="*70)
            print("  👋 ¡Gracias por usar Toca y Corre!")
            print("     Sistema de Aprendizaje por Refuerzo")
            print("     Desarrollado por: Ariel Huarachi Clemente")
            print("="*70 + "\n")
            break

        else:
            print("\n❌ Opción inválida. Por favor elige 1-6\n")


# ==========================================
#   EJECUCIÓN
# ==========================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Programa interrumpido por el usuario")
        print("👋 ¡Hasta luego!\n")
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        print("Por favor reporta este error.\n")