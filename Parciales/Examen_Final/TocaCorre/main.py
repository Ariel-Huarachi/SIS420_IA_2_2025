import numpy as np
from train import entrenar, guardar_estadisticas, cargar_estadisticas
from collections import defaultdict
import os

# ==========================================
#   GUARDAR/CARGAR MODELO
# ==========================================

def guardar_modelo(Q_agentes, stats=None):
    """Guarda modelo y estadísticas"""
    try:
        for i, Q in enumerate(Q_agentes):
            np.save(f"Q_agente_{i}.npy", dict(Q))
        
        print("\n✅ Modelo guardado exitosamente")
        for i, Q in enumerate(Q_agentes):
            print(f"   - Q_agente_{i}.npy ({len(Q)} estados)")

        if stats is not None:
            guardar_estadisticas(stats, "training_stats_3agentes.npy")

        return True
    except Exception as e:
        print(f"\n❌ Error al guardar: {e}")
        return False


def cargar_modelo():
    """Carga modelo entrenado"""
    try:
        Q_agentes = []
        for i in range(3):
            Q_dict = np.load(f"Q_agente_{i}.npy", allow_pickle=True).item()
            Q = defaultdict(lambda: np.zeros(9))
            Q.update(Q_dict)
            Q_agentes.append(Q)

        print("\n✅ Modelo cargado correctamente")
        for i, Q in enumerate(Q_agentes):
            print(f"   Estados Agente {i}: {len(Q)}")
        
        return Q_agentes
    except FileNotFoundError:
        print("\n❌ No se encontró modelo entrenado")
        print("   Por favor entrena primero (opción 1)")
        return None
    except Exception as e:
        print(f"\n❌ Error al cargar: {e}")
        return None


def verificar_modelo_existe():
    """Verifica si existe un modelo guardado"""
    return all(os.path.exists(f"Q_agente_{i}.npy") for i in range(3))


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
        print("\nℹ️  Entrada inválida, usando valores por defecto\n")
        return 5000, 0.95, 0.1


def analizar_estadisticas():
    """Analiza y muestra estadísticas del entrenamiento"""
    stats = cargar_estadisticas()
    
    if stats is None:
        print("❌ No hay estadísticas disponibles\n")
        return
    
    print("\n" + "="*70)
    print("  📊 ANÁLISIS DE ESTADÍSTICAS DE ENTRENAMIENTO")
    print("="*70)
    
    print(f"\n📈 Recompensas por agente:")
    for i in range(3):
        if len(stats['recompensas_agente'][i]) > 0:
            rews = stats['recompensas_agente'][i]
            print(f"\n   Agente {i}:")
            print(f"     - Promedio: {np.mean(rews):.1f}")
            print(f"     - Máximo: {np.max(rews):.1f}")
            print(f"     - Mínimo: {np.min(rews):.1f}")
    
    if len(stats['pasos_por_episodio']) > 0:
        print(f"\n🚶 Pasos por episodio:")
        pasos = stats['pasos_por_episodio']
        print(f"   - Promedio: {np.mean(pasos):.1f}")
        print(f"   - Máximo: {np.max(pasos)}")
        print(f"   - Mínimo: {np.min(pasos)}")
    
    if len(stats['capturas_totales']) > 0:
        print(f"\n🎯 Capturas totales:")
        capturas = stats['capturas_totales']
        print(f"   - Promedio por episodio: {np.mean(capturas):.2f}")
        print(f"   - Total acumulado: {np.sum(capturas)}")
    
    print("\n" + "="*70 + "\n")


# ==========================================
#   MENÚ PRINCIPAL
# ==========================================

def main():
    """Menú principal del juego"""
    print("\n" + "="*70)
    print("      🎮 TOCA Y CORRE - 3 AGENTES EN CICLO")
    print("        Aprendizaje por Refuerzo")
    print("        Estructura: A->B->C->A")
    print("="*70)

    while True:
        modelo_existe = verificar_modelo_existe()
        
        print("\n📋 MENÚ PRINCIPAL")
        print("-" * 50)
        print("  1️⃣  Entrenar agentes desde cero")
        print("  2️⃣  Analizar estadísticas" + 
              (" ✅" if os.path.exists("training_stats_3agentes.npy") else " ⚠️"))
        print("  3️⃣  Continuar entrenamiento existente")
        print("  4️⃣  Salir")
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
            Q_agentes, stats = entrenar(
                q_episodes=ep, gamma=gamma, alpha=alpha, verbose=True
            )

            guardar_modelo(Q_agentes, stats)
            print("\n✅ ¡Entrenamiento completado exitosamente!\n")
            
            input("Presiona ENTER para continuar...")

        # ============ OPCIÓN 2: ANALIZAR ============
        elif opcion == "2":
            analizar_estadisticas()
            input("Presiona ENTER para continuar...")

        # ============ OPCIÓN 3: CONTINUAR ENTRENAMIENTO ============
        elif opcion == "3":
            print("\n" + "="*70)
            print("  ♻️  CONTINUAR ENTRENAMIENTO EXISTENTE")
            print("="*70)

            Q_agentes = cargar_modelo()
            if Q_agentes is None:
                print("\n⚠️  No hay modelo previo. Usa opción 1 para entrenar desde cero\n")
                input("Presiona ENTER para continuar...")
                continue

            ep = input("\nEpisodios adicionales [2000]: ") or "2000"
            ep = int(ep)
            
            print(f"\n🚀 Continuando entrenamiento por {ep} episodios...\n")
            
            Q_agentes, stats = entrenar(
                q_episodes=ep, gamma=0.95, alpha=0.05, verbose=True
            )
            
            guardar_modelo(Q_agentes, stats)
            print("\n✅ Entrenamiento adicional completado\n")
            
            input("Presiona ENTER para continuar...")

        # ============ OPCIÓN 4: SALIR ============
        elif opcion == "4":
            print("\n" + "="*70)
            print("  👋 ¡Gracias por usar Toca y Corre 3 Agentes!")
            print("     Sistema de Aprendizaje por Refuerzo")
            print("="*70 + "\n")
            break

        else:
            print("\n❌ Opción inválida. Por favor elige 1-4\n")


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